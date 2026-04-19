# PRD-PACK-032: Building Energy Assessment Pack

**Pack ID:** PACK-032-building-energy-assessment
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-20
**Prerequisite:** None (standalone; enhanced with PACK-031 Industrial Energy Audit Pack if present; complemented by PACK-021/022/023 Net Zero Packs)

---

## 1. Executive Summary

### 1.1 Problem Statement

Buildings account for approximately 40% of total energy consumption and 36% of energy-related CO2 emissions in the European Union (European Commission, EU Building Stock Observatory 2024). The recast Energy Performance of Buildings Directive (EPBD) -- Directive (EU) 2024/1275 -- establishes a clear trajectory toward a zero-emission building stock by 2050, with mandatory Minimum Energy Performance Standards (MEPS), enhanced Energy Performance Certificate (EPC) requirements, and progressive renovation obligations. Despite this comprehensive regulatory framework, building owners, operators, and energy assessors face significant operational challenges:

1. **EPC generation complexity and inconsistency**: Energy Performance Certificates are mandatory across EU Member States for building sale, rental, and major renovation, yet EPC methodologies vary significantly between nations (SAP/RdSAP in the UK, EnEV/GEG in Germany, DPE in France, APE in Italy, CALENER in Spain). Each methodology involves hundreds of input parameters covering building geometry, construction materials, HVAC systems, lighting, domestic hot water, and renewable energy systems. A single EPC for a commercial building requires 40-80 hours of assessment work by a qualified assessor, costing EUR 2,000-8,000. Inconsistencies between assessors for the same building can vary by 1-2 letter grades, undermining market confidence.

2. **Building envelope thermal performance assessment**: Calculating U-values for walls, roofs, floors, and windows per EN ISO 6946 requires detailed knowledge of construction layers, material thermal conductivities, surface resistances, and thermal bridging corrections. Thermal bridge psi-values per EN ISO 10211 require 2D/3D thermal modelling that most assessors cannot perform. Air tightness (n50, q50 per EN ISO 9972) and condensation risk (Glaser method per EN ISO 13788) add further analytical complexity. Many assessments use default values that overestimate thermal transmittance by 20-50%, resulting in artificially poor EPC ratings.

3. **HVAC system efficiency quantification**: Modern buildings employ increasingly complex HVAC configurations -- heat pumps with supplementary boilers, variable refrigerant flow (VRF) systems, district heating connections, mechanical ventilation with heat recovery (MVHR), mixed-mode ventilation, and hybrid systems combining multiple technologies. Quantifying system efficiency requires seasonal performance factors (SPF for heat pumps per EN 14825, SEER/EER for cooling per EN 14511), distribution losses, control effectiveness, and auxiliary energy consumption. Most assessors default to generic efficiency values that fail to capture actual system performance.

4. **Regulatory compliance fragmentation**: Building owners face a patchwork of overlapping regulations -- EPBD for energy performance certification, Minimum Energy Efficiency Standards (MEES) for rental properties (UK/EU), Building Performance Standards (BPS) under Fit for 55, F-gas Regulation for refrigerant tracking, Part L/F Building Regulations (UK), EN 16798-1 for indoor environment parameters, and national building codes. No integrated tool maps a building's current performance against all applicable regulatory requirements simultaneously, leaving compliance gaps undetected.

5. **Retrofit decision-making complexity**: Building retrofit involves 60+ potential measures spanning envelope (insulation, glazing, air tightness), HVAC (heat pump conversion, controls upgrade, heat recovery), lighting (LED, controls, daylighting), renewables (solar PV, solar thermal), and demand response. Measure interactions are complex -- envelope improvements reduce heating demand, changing the optimal heat pump sizing; LED lighting reduces internal heat gains, increasing heating demand in winter but reducing cooling in summer. Without interaction-aware modelling, retrofit plans either underestimate or overestimate savings by 15-30%.

6. **Nearly Zero-Energy Building (nZEB) gap analysis**: The EPBD requires all new buildings to be zero-emission buildings (ZEB) from 2030, and all existing buildings undergoing major renovation to meet nZEB standards. However, the gap between current building performance (typically EPC D-G) and nZEB/ZEB requirements (EPC A or equivalent) is substantial. Quantifying this gap and defining a staged retrofit pathway to bridge it -- with cost-benefit analysis at each stage -- requires sophisticated energy modelling that exceeds typical assessment capabilities.

7. **Whole life carbon blind spots**: EN 15978:2011 defines lifecycle stages for construction works (A1-A5 production and construction, B1-B7 use phase, C1-C4 end-of-life, D beyond the system boundary), yet most building assessments focus exclusively on operational carbon (B6 operational energy use). Embodied carbon from materials (A1-A3), construction (A4-A5), replacement (B4), and end-of-life (C1-C4) can represent 30-70% of whole life carbon for new low-energy buildings. Without whole life carbon assessment, retrofit decisions may reduce operational carbon while inadvertently increasing embodied carbon, producing suboptimal climate outcomes.

8. **CRREM stranding risk**: The Carbon Risk Real Estate Monitor (CRREM) provides science-based decarbonization pathways for 60+ building types across 40+ countries. Buildings that exceed their CRREM pathway become "stranded assets" -- properties that face regulatory restrictions, reduced market value, and inability to attract tenants or financing. Most building owners lack tools to assess their CRREM pathway position and project when (or whether) their building will become stranded under business-as-usual scenarios.

9. **Certification complexity (LEED/BREEAM/Energy Star)**: Green building certification schemes -- LEED v4.1, BREEAM, Energy Star, DGNB, HQE -- each have distinct energy credit requirements with detailed calculation methodologies. LEED EA credits require ASHRAE 90.1 baseline comparison with percentage improvement. BREEAM Ene credits require EPBD compliance plus additional performance thresholds. Energy Star requires normalized source EUI at the 75th percentile. Pursuing multiple certifications simultaneously requires redundant assessments with different methodologies, multiplying cost and effort.

10. **Indoor environment quality trade-offs**: EN 16798-1:2019 defines four indoor environment categories (I-IV) for thermal comfort, indoor air quality, lighting, and acoustics. Achieving higher indoor environment quality often conflicts with energy efficiency -- higher ventilation rates for better air quality increase heating/cooling demand; larger windows for better daylighting increase solar gains and thermal bridging. Optimizing the energy-IEQ trade-off requires integrated assessment that balances thermal comfort (PMV/PPD per ISO 7730), air quality (CO2, PM2.5, VOC), daylighting (EN 17037), and overheating risk (TM59) against energy consumption.

### 1.2 Solution Overview

PACK-032 is the **Building Energy Assessment Pack** -- the second pack in the "Energy Efficiency Packs" category, complementing PACK-031 (Industrial Energy Audit) which covers industrial facilities. PACK-032 provides a comprehensive building energy performance assessment solution purpose-built for EPBD-compliant EPC generation, EN ISO 52000-1 energy performance calculation, building envelope thermal analysis, HVAC system assessment, retrofit planning, benchmarking, green building certification support, whole life carbon assessment, CRREM pathway compliance, and indoor environment quality evaluation.

The pack covers the full building energy assessment lifecycle: initial building survey, EPC generation, HVAC system analysis, domestic hot water assessment, lighting evaluation, renewable energy integration, performance benchmarking, deep retrofit planning, indoor environment assessment, whole life carbon calculation, and ongoing compliance monitoring.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete building energy assessment and management lifecycle for commercial, residential, and public sector buildings of all types and sizes.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-032 Building Energy Assessment Pack |
|-----------|------------------------------|------------------------------------------|
| Time to complete EPC (commercial) | 40-80 hours | <4 hours (10-20x faster) |
| EPC assessment cost | EUR 2,000-8,000 per building | EUR 200-800 per building (10x reduction) |
| U-value calculation accuracy | Default values (20-50% overestimate) | Calculated per EN ISO 6946 with thermal bridging corrections |
| HVAC efficiency assessment | Generic default efficiencies | Seasonal performance calculation per EN 14825/14511 |
| Retrofit measure analysis | 10-15 measures considered | 60+ measures with interaction modelling |
| nZEB gap analysis | Qualitative assessment | Quantified gap with staged retrofit pathway and cost-benefit |
| CRREM pathway compliance | Manual annual check | Automated pathway tracking with stranding year projection |
| Whole life carbon | Rarely assessed | Full EN 15978 lifecycle (A1-D) with material substitution analysis |
| Green building certification | Separate consultant per scheme | Integrated LEED/BREEAM/Energy Star scoring |
| Indoor environment quality | Separate comfort study | Integrated PMV/PPD, IAQ, daylighting, and overheating assessment |
| Audit trail | Paper-based assessment files | SHA-256 provenance, full calculation lineage, digital audit trail |
| Multi-country EPC | Country-specific assessor required | Configurable national methodology (SAP, GEG, DPE, APE, etc.) |

### 1.4 Building Types Covered

| Building Category | Subtypes | Key Energy Characteristics |
|-------------------|----------|---------------------------|
| Commercial Office | Open-plan, cellular, mixed, high-rise | Cooling-dominated (internal gains), lighting 20-30%, IT loads, occupancy-driven HVAC |
| Retail | Shopping centre, standalone, food retail | Refrigeration (food retail 40-60%), lighting 25-35%, long operating hours, high ventilation |
| Hotel / Hospitality | Business, resort, serviced apartments | DHW 20-30%, heating/cooling, 24/7 operation, variable occupancy, kitchen energy |
| Healthcare | Hospital, clinic, care home | 24/7 operation, high ventilation (infection control), sterilization, medical equipment, DHW |
| Education | School, university, nursery | Intermittent occupancy, high ventilation per pupil, heating-dominated, sports facilities |
| Residential Multifamily | Apartment block, social housing, co-living | Heating 50-70%, DHW 15-25%, individual metering, common area energy |
| Mixed-Use Development | Residential + commercial, live-work | Multiple use zones, shared plant, complex metering, diverse schedules |
| Public Sector | Government, library, museum, community | Heritage constraints, variable hours, public access, display/storage requirements |

### 1.5 Target Users

**Primary:**
- Building energy assessors conducting EPC assessments (commercial and domestic)
- Facilities managers responsible for building energy performance
- Property portfolio managers tracking CRREM pathway compliance and stranding risk
- Building owners planning major renovation or retrofit

**Secondary:**
- Architects and building services engineers designing new low-energy buildings
- Local authority building control officers verifying Part L/EPBD compliance
- Green building certification consultants (LEED AP, BREEAM Assessor)
- Real estate investors evaluating energy risk in property portfolios
- Housing associations managing residential stock energy performance
- Corporate sustainability teams reporting Scope 1+2 building emissions
- Tenant energy managers monitoring operational performance
- ESCOs developing building energy performance contracts

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to complete EPC assessment (commercial building) | <4 hours (vs. 40-80 manual) | Time from building data input to EPC report |
| U-value calculation accuracy | Within 5% of measured values (heat flux meter) | Validated against 200 measured U-values |
| EPC rating accuracy | 100% match with national calculation engine (SAP, GEG, etc.) | Cross-validated against official calculation tools |
| HVAC seasonal efficiency calculation | Within 3% of monitored performance | Validated against BMS-monitored seasonal data |
| Retrofit savings prediction accuracy | Within 15% of post-retrofit measured performance | M&V validation at 12 months post-retrofit |
| CRREM pathway assessment accuracy | 100% match with CRREM online tool | Cross-validated against CRREM v2.0 output |
| LEED/BREEAM credit estimation | Within 1 credit of official assessment | Validated against 50 certified projects |
| Whole life carbon calculation | Within 10% of detailed LCA (One Click LCA, etc.) | Cross-validated against full LCA tool output |
| Indoor environment assessment | 100% compliance with EN 16798-1 methodology | Validated against detailed IEQ simulation |
| Customer NPS | >50 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EU EPBD (Energy Performance of Buildings Directive) | Directive (EU) 2024/1275 (recast) | Core regulatory driver; EPC requirements, MEPS, zero-emission building targets, renovation obligations |
| EN ISO 52000-1:2017 | Energy performance of buildings -- Overarching EPB assessment | Framework standard linking all EPB calculation standards; defines overall energy performance assessment methodology |
| EN ISO 52016-1:2017 | Energy needs for heating and cooling, internal temperatures, sensible and latent heat loads | Hourly/monthly calculation of heating and cooling energy needs from building physics |
| EN ISO 52003-1:2017 | Energy performance of buildings -- Indicators for partial EPB requirements related to thermal energy balance and fabric features | EPC indicator definitions, classification framework, rating methodology |
| EN 15603:2008 | Energy performance of buildings -- Overall energy use and definition of energy ratings | Primary energy calculation methodology, energy balance, delivered/exported energy |
| EN 15978:2011 | Sustainability of construction works -- Assessment of environmental performance of buildings | Whole life carbon assessment framework (lifecycle stages A1-D) |
| EN 16798-1:2019 | Energy performance of buildings -- Ventilation for buildings -- Indoor environmental input parameters | Indoor environment categories (I-IV), ventilation rates, thermal comfort, air quality parameters |
| ISO 7730:2005 | Ergonomics of the thermal environment -- Analytical determination of thermal comfort using PMV and PPD | Thermal comfort calculation (PMV/PPD, local discomfort) |

### 2.2 Building Physics Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EN ISO 6946:2017 | Building components and building elements -- Thermal resistance and thermal transmittance -- Calculation method | U-value calculation for opaque elements (walls, roofs, floors) |
| EN ISO 10211:2017 | Thermal bridges in building construction -- Heat flows and surface temperatures | Linear and point thermal bridge psi-value and chi-value calculation |
| EN ISO 13788:2012 | Hygrothermal performance of building components -- Internal surface temperature to avoid critical surface humidity and interstitial condensation | Condensation risk assessment (Glaser method) |
| EN ISO 9972:2015 | Thermal performance of buildings -- Determination of air permeability of buildings -- Fan pressurization method | Air tightness measurement methodology (n50, q50, w50) |
| EN ISO 13370:2017 | Thermal performance of buildings -- Heat transfer via the ground | Ground floor U-value calculation with ground coupling |
| EN ISO 10077-1:2017 | Thermal performance of windows, doors and shutters -- Calculation of thermal transmittance | Window Uw-value calculation from frame, glazing, and spacer |
| EN 14351-1:2006+A2:2016 | Windows and doors -- Product standard, performance characteristics | Window performance (Uw, g-value, air permeability class) |
| EN 17037:2018 | Daylight in buildings | Daylight factor assessment, view out, sun exposure |

### 2.3 HVAC and Services Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EN 14825:2022 | Air conditioners, liquid chilling packages, heat pumps -- Testing and rating at part load conditions | Heat pump seasonal COP (SCOP) and cooling SEER calculation |
| EN 14511-1:2022 | Air conditioners, liquid chilling packages and heat pumps -- Terms, definitions and classifications | Full load EER/COP testing and rating |
| EN 15316 series | Energy performance of buildings -- Method for calculation of system energy requirements and system efficiencies | Heating (Part 2), DHW (Part 3), space cooling (Part 3-1), lighting (Part 6-1) |
| EN 15193-1:2017 | Energy performance of buildings -- Energy requirements for lighting -- Part 1: Specifications | LENI (Lighting Energy Numeric Indicator) calculation |
| EN 12464-1:2021 | Light and lighting -- Lighting of work places -- Part 1: Indoor work places | Lighting power density benchmarks per space type |
| EN 15232-1:2017 | Energy performance of buildings -- Impact of building automation, controls and building management | BACS efficiency factors for building controls classification (A-D) |

### 2.4 Green Building Certification Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| ASHRAE 90.1-2022 | Energy Standard for Buildings Except Low-Rise Residential Buildings | Baseline building model for LEED energy credit, prescriptive/performance path requirements |
| LEED v4.1 | Leadership in Energy and Environmental Design | Energy credits (EA Prerequisite, EA Credit), IEQ credits, materials credits |
| BREEAM International 2024 | Building Research Establishment Environmental Assessment Method | Ene credits (energy performance, reduction of CO2 emissions), Mat credits (lifecycle impact) |
| Energy Star Portfolio Manager | US EPA Building Benchmarking | Normalized source EUI calculation, 1-100 Energy Star score, certification threshold (75+) |
| DGNB Certification | German Sustainable Building Council | ENV criteria (lifecycle environmental impact), ECO criteria (lifecycle costs) |
| NABERS | National Australian Built Environment Rating System | Operational energy and water rating |

### 2.5 Decarbonization and Pathway Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| CRREM (Carbon Risk Real Estate Monitor) | CRREM v2.0 (2023, updated annually) | Science-based decarbonization pathways for 60+ building types across 40+ countries; stranding year calculation |
| RIBA 2030 Climate Challenge | RIBA (2021) | Operational and embodied carbon targets per building type |
| LETI Climate Emergency Design Guide | LETI (2020) | Whole life carbon budgets, space heating demand targets, energy use intensity targets |
| UK Net Zero Carbon Buildings Standard | UKGBC (2024) | Net zero carbon building definition, verification methodology |
| EU Building Performance Standards (BPS) | Fit for 55 package | Minimum performance thresholds with timeline for worst-performing buildings |
| Passive House Standard | PHI (Passivhaus Institut) | Space heating demand <= 15 kWh/m2/yr, air tightness n50 <= 0.6 ACH, primary energy <= 120 kWh/m2/yr |

### 2.6 National EPC Methodologies

| Country | Methodology | Reference | Key Differences |
|---------|-------------|-----------|-----------------|
| UK | SAP 10.2 / RdSAP | BRE (2022) | Carbon-based rating, fabric energy efficiency, dwelling emission rate vs. target |
| Germany | GEG (Gebaudeenergiegesetz) | GEG 2024 | Primary energy demand + CO2 emissions, reference building method, Energieausweis |
| France | DPE (Diagnostic de Performance Energetique) | 3CL-DPE 2021 | Dual rating (energy + GHG), unified calculation, 5-usage scope (heating, cooling, DHW, lighting, auxiliary) |
| Italy | APE (Attestato di Prestazione Energetica) | UNI/TS 11300 series | Non-renewable primary energy ratio, renewable energy contribution, 10-class system (A4-G) |
| Spain | CALENER | CTE 2019 (Codigo Tecnico de la Edificacion) | CO2 emissions index, non-renewable primary energy, heating/cooling demand limits |
| Netherlands | NTA 8800 | NEN (2020) | BENG indicators (energy demand, primary energy, renewable share), label A++++-G |
| Belgium | EPB (Energie Prestatie en Binnenklimaat) | Regional (Flanders/Wallonia/Brussels) | E-level (global insulation), S-level (heating demand), three regional methods |
| Ireland | BER (Building Energy Rating) | SEAI DEAP 4.2 | Energy Performance Coefficient, Carbon Performance Coefficient, primary energy |
| Austria | OIB RL 6 | OIB (2023) | Heating demand (HWB), primary energy (PEB), CO2 emissions (CO2), total energy efficiency factor |
| Sweden | BEN | Boverket (2024) | Primary energy number, building category factors, climate zone weighting |

### 2.7 Supporting Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 1+2 emissions from building energy consumption |
| ISO 14064-1:2018 | Organization GHG quantification | Building-related GHG emissions quantification |
| ESRS E1 Climate Change | EU CSRD (2023) | E1-5 energy consumption and mix disclosure for building portfolios |
| EN 15804:2012+A2:2019 | Sustainability of construction works -- Environmental product declarations (EPD) | EPD data for whole life carbon calculation (embodied carbon) |
| ISO 14040/14044 | Environmental management -- Life cycle assessment | LCA methodology framework for whole life carbon |
| EU Taxonomy Regulation | Regulation 2020/852 | Climate mitigation substantial contribution criteria for building renovation activities |
| F-gas Regulation | Regulation (EU) 2024/573 | Refrigerant tracking, GWP phase-down schedule, leak checking requirements |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Building energy assessment calculation engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report, dashboard, certificate, and compliance templates |
| Integrations | 12 | Agent, app, data, certification, and system bridges |
| Presets | 8 | Building-type-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `building_envelope_engine.py` | Calculates thermal performance of the building envelope: U-values for walls, roofs, and floors per EN ISO 6946, ground floor heat transfer per EN ISO 13370, window thermal transmittance (Uw) per EN ISO 10077-1, solar heat gain coefficient (g-value) per EN 14351-1, linear thermal bridge psi-values per EN ISO 10211, air tightness metrics (n50, q50) per EN ISO 9972, condensation risk assessment via Glaser method per EN ISO 13788, thermal mass and decrement factor calculation. |
| 2 | `epc_rating_engine.py` | Generates Energy Performance Certificate ratings A-G per EPBD methodology: primary energy calculation per EN 15603, CO2 emissions rating, reference building comparison, energy needs calculation per EN ISO 52016-1, delivered energy by carrier, exported energy credits, renewable energy contribution, national methodology variants (SAP for UK, GEG for DE, DPE for FR, APE for IT, NTA 8800 for NL, CALENER for ES). |
| 3 | `hvac_assessment_engine.py` | Assesses heating, ventilation, and air conditioning system efficiency: heating system seasonal efficiency (AFUE for boilers, SCOP for heat pumps per EN 14825), cooling system seasonal efficiency (SEER/EER per EN 14511), heat pump SPF calculation across system boundaries (SPF1-SPF4), district heating/cooling assessment, ventilation heat recovery effectiveness per EN 308, building automation and controls efficiency per EN 15232, refrigerant tracking with F-gas Regulation compliance. |
| 4 | `domestic_hot_water_engine.py` | Calculates domestic hot water energy demand and system efficiency: DHW demand per EN 15316-3 by building type and occupancy, system efficiency assessment including generation (boiler, heat pump, direct electric, solar thermal), storage losses (cylinder standing loss), distribution losses (primary and secondary pipework), solar thermal contribution via f-chart method, heat pump water heater COP at varying inlet/outlet temperatures, legionella compliance (weekly pasteurization cycle energy), and point-of-use vs. centralized system comparison. |
| 5 | `lighting_assessment_engine.py` | Evaluates lighting energy performance: Lighting Power Density (LPD) benchmarking per EN 12464-1 required illuminance levels, LENI (Lighting Energy Numeric Indicator) calculation per EN 15193-1, daylight factor assessment per EN 17037, daylight autonomy and useful daylight illuminance metrics, lighting control credits (occupancy/absence detection, daylight dimming, time scheduling, personal control), circadian lighting quality (melanopic equivalent daylight illuminance), emergency lighting energy, and external lighting assessment. |
| 6 | `renewable_integration_engine.py` | Assesses on-site and near-site renewable energy potential: solar PV sizing and yield estimation (kWp capacity, annual kWh/kWp by location, orientation, tilt, shading), solar thermal contribution per EN 15316-4-3 (collector efficiency, storage sizing, solar fraction), building-integrated photovoltaics (BIPV) assessment for facades and roofs, building-integrated solar thermal, small wind potential, ground/air source heat pump sizing, renewable energy fraction calculation per EPBD, on-site vs. off-site generation comparison, PPA (Power Purchase Agreement) assessment for off-site renewables. |
| 7 | `building_benchmark_engine.py` | Benchmarks building energy performance against peers: EUI (Energy Use Intensity, kWh/m2/yr) calculation by end use and total, weather-normalized EUI using degree-day regression, Energy Star score estimation (1-100) using CBECS regression models, CRREM pathway compliance check with stranding year projection, DEC (Display Energy Certificate) operational rating, CIBSE TM46 energy benchmarks by building type, sector peer comparison by climate zone, NABERS star rating estimation, and trend analysis with year-over-year tracking. |
| 8 | `retrofit_analysis_engine.py` | Analyzes building retrofit opportunities: comprehensive measure library (60+ measures across envelope, HVAC, lighting, renewables, controls, demand response), measure interaction modelling (heating/cooling load interaction, lighting-HVAC interaction, envelope-HVAC sizing interaction), cost-benefit analysis per measure and package (NPV, IRR, simple payback, lifecycle cost), staged retrofit roadmap from current performance to nZEB/ZEB target, nZEB gap analysis quantifying distance to nearly zero-energy performance, EnerPHit (Passive House retrofit) compliance check, financing options assessment (grants, green loans, ESCO model, energy performance contracts), and sensitivity analysis on energy price, discount rate, and climate scenarios. |
| 9 | `indoor_environment_engine.py` | Evaluates indoor environmental quality: thermal comfort via PMV/PPD calculation per ISO 7730 (6-factor model: air temperature, mean radiant temperature, relative humidity, air velocity, metabolic rate, clothing insulation), adaptive thermal comfort per EN 16798-1 (running mean outdoor temperature method), indoor air quality assessment (CO2 concentration, PM2.5, PM10, TVOC, formaldehyde levels), ventilation adequacy per EN 16798-1 (category I-IV minimum rates per person and per m2), overheating risk assessment per CIBSE TM59 (criterion A: living rooms/bedrooms <= 3% hours above comfort threshold; criterion B: bedrooms <= 1% hours above 26C), and daylighting quality per EN 17037 (target illuminance, uniformity, view out). |
| 10 | `whole_life_carbon_engine.py` | Calculates whole life carbon per EN 15978 lifecycle stages: embodied carbon (A1-A3 product stage from EPD data, A4 transport, A5 construction), use stage carbon (B1 use/emissions, B2 maintenance, B3 repair, B4 replacement, B5 refurbishment, B6 operational energy use, B7 operational water use), end-of-life carbon (C1 deconstruction, C2 transport, C3 waste processing, C4 disposal), beyond system boundary (D reuse/recovery/recycling benefits), whole life carbon budget comparison per RIBA/LETI targets, material substitution analysis (e.g., concrete to timber, steel to CLT), biogenic carbon accounting per EN 16449, and upfront carbon (A1-A5) vs. operational carbon (B6) trade-off analysis. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `initial_building_assessment_workflow.py` | 5: BuildingRegistration -> DataCollection -> EnvelopeAnalysis -> SystemsAssessment -> PerformanceReport | End-to-end initial building energy assessment from building setup to comprehensive performance report |
| 2 | `epc_generation_workflow.py` | 4: BuildingGeometry -> FabricCalculation -> SystemsCalculation -> EPCIssuance | Energy Performance Certificate generation workflow per national methodology |
| 3 | `retrofit_planning_workflow.py` | 4: BaselinePerformance -> MeasureIdentification -> PackageOptimization -> RoadmapGeneration | Deep retrofit planning from current performance assessment to staged implementation roadmap |
| 4 | `continuous_building_monitoring_workflow.py` | 4: RealTimeDataIngestion -> PerformanceTracking -> DeviationAlerts -> TrendReporting | Ongoing building energy performance monitoring with anomaly detection |
| 5 | `certification_assessment_workflow.py` | 4: SchemeSelection -> CreditMapping -> PerformanceCalculation -> CertificationScorecard | Green building certification assessment for LEED, BREEAM, Energy Star, DGNB |
| 6 | `tenant_engagement_workflow.py` | 3: TenantMetering -> ConsumptionReporting -> BehaviourRecommendations | Tenant-facing energy reporting with consumption breakdown and reduction guidance |
| 7 | `regulatory_compliance_workflow.py` | 3: ObligationMapping -> ComplianceAssessment -> DeadlineTracking | EPBD/MEES/BPS regulatory compliance management with multi-jurisdiction support |
| 8 | `nzeb_readiness_workflow.py` | 4: CurrentPerformance -> GapQuantification -> RetrofitPathway -> ZEBRoadmap | Nearly Zero-Energy Building / Zero-Emission Building readiness assessment and pathway |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `epc_report.py` | MD, HTML, PDF, JSON | Energy Performance Certificate report per EPBD with A-G rating, primary energy, CO2 emissions, recommendations |
| 2 | `dec_report.py` | MD, HTML, PDF, JSON | Display Energy Certificate for operational rating based on metered energy consumption |
| 3 | `building_assessment_report.py` | MD, HTML, PDF, JSON | Comprehensive building energy assessment report covering envelope, systems, benchmarks, and recommendations |
| 4 | `retrofit_recommendation_report.py` | MD, HTML, PDF, JSON | Retrofit business case with measure details, savings, costs, NPV/IRR, staged roadmap |
| 5 | `building_benchmark_report.py` | MD, HTML, JSON | Peer comparison dashboard with EUI benchmarks, Energy Star score, CRREM pathway position |
| 6 | `certification_scorecard.py` | MD, HTML, PDF, JSON | LEED/BREEAM/Energy Star scorecard with credit-by-credit assessment and gap-to-certification |
| 7 | `tenant_energy_report.py` | MD, HTML, PDF, JSON | Tenant-facing energy report with consumption breakdown, cost allocation, and reduction tips |
| 8 | `building_dashboard.py` | MD, HTML, JSON | Real-time building performance dashboard with EUI tracking, system efficiency, alerts |
| 9 | `regulatory_compliance_report.py` | MD, HTML, PDF, JSON | EPBD/MEES/BPS compliance summary with obligation status, deadlines, and remediation actions |
| 10 | `whole_life_carbon_report.py` | MD, HTML, PDF, JSON | Embodied + operational carbon report per EN 15978 with lifecycle stage breakdown and budget comparison |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `building_assessment_orchestrator.py` | 12-phase DAG pipeline with retry, provenance, conditional assessment phases (DHW, lighting, renewables, certification based on building type and scope) |
| 2 | `mrv_building_bridge.py` | Routes to 8 building-specific MRV agents (GL-MRV-BLD-001 through GL-MRV-BLD-008) for building energy-related GHG emissions (Scope 1 gas/oil combustion, Scope 2 purchased electricity/heat/cooling, refrigerant leakage) |
| 3 | `data_building_bridge.py` | Routes to DATA agents for building data ingestion: DATA-002 (Excel/CSV for utility bills and BMS exports), DATA-001 (PDF extraction for EPC certificates and survey forms), DATA-003 (ERP/finance for energy procurement), DATA-010 (data quality profiling) |
| 4 | `epbd_compliance_bridge.py` | EPBD Directive (EU) 2024/1275 compliance integration: EPC validity tracking, MEPS threshold monitoring, renovation obligation assessment, zero-emission building timeline compliance, Building Renovation Passport support |
| 5 | `bms_integration_bridge.py` | Building Management System / Building Automation and Controls System data integration: real-time energy data, HVAC setpoints, zone temperatures, equipment status via BACnet/IP, Modbus TCP/RTU, KNX, LonWorks, OPC-UA protocols |
| 6 | `weather_data_bridge.py` | Weather data integration for energy calculation and normalization: TMY (Typical Meteorological Year) data for design calculations, degree-day data (HDD/CDD) for benchmarking normalization, solar radiation for PV yield and solar thermal, climate zone classification |
| 7 | `certification_bridge.py` | Green building certification API integration: LEED Online credit tracking, BREEAM scheme data, Energy Star Portfolio Manager API, DGNB certification data exchange, certification pre-assessment validation |
| 8 | `grid_carbon_bridge.py` | Real-time and annual grid carbon intensity data: national grid emission factors (DEFRA, UBA, ADEME, ISPRA), half-hourly marginal emission factors for demand response, renewable penetration forecasting, residual mix factors for Scope 2 market-based reporting |
| 9 | `property_registry_bridge.py` | Building and property database integration: cadastral data (building footprint, height, year of construction), EPC register lookup (existing certificates, lodgement history), Land Registry data, building control approvals, listed building/heritage constraints |
| 10 | `health_check.py` | 20-category system verification covering all 10 engines, 8 workflows, BMS connectivity, weather data freshness, certification API status, and database health |
| 11 | `setup_wizard.py` | 8-step guided building configuration: building profile (type, age, area, floors), envelope construction, HVAC systems, lighting, DHW, renewables, metering infrastructure, regulatory jurisdiction |
| 12 | `crrem_pathway_bridge.py` | CRREM (Carbon Risk Real Estate Monitor) integration: pathway data for 60+ building types across 40+ countries, annual carbon intensity threshold lookup, stranding year calculation, scenario analysis (1.5C, 2.0C pathways), retrofit impact on pathway compliance |

### 3.6 Presets

| # | Preset | Building Type | Key Characteristics |
|---|--------|--------------|---------------------|
| 1 | `commercial_office.yaml` | Commercial Office | Cooling-dominated in summer, internal gains from IT/lighting/occupancy, typical EUI 150-300 kWh/m2/yr, HVAC 40-60% of energy, lighting 20-30%, benchmarked against CIBSE TM46 Type 1 |
| 2 | `retail_building.yaml` | Retail / Shopping Centre | Long operating hours (12-16 hrs/day), high lighting density (15-25 W/m2), refrigeration for food retail, high ventilation for air curtains, benchmarked against CIBSE TM46 Type 2 |
| 3 | `hotel_hospitality.yaml` | Hotel / Hospitality | 24/7 operation, high DHW demand (100-200 L/room/day), variable occupancy, kitchen energy 15-25%, laundry energy, guest room HVAC controls, benchmarked against CIBSE TM46 Type 7 |
| 4 | `healthcare_facility.yaml` | Healthcare (Hospital/Clinic) | 24/7 operation, strict ventilation (6-15 ACH for theatres/isolation), sterilization and autoclaving, medical equipment loads, emergency power, benchmarked against CIBSE TM46 Type 3/4 |
| 5 | `education_building.yaml` | Education (School/University) | Intermittent occupancy (term-time only), high ventilation per pupil (8-10 L/s per person), heating-dominated, sports halls/pools, IT suites, benchmarked against CIBSE TM46 Type 5 |
| 6 | `residential_multifamily.yaml` | Residential Multifamily | Heating-dominated (50-70%), DHW 15-25%, individual unit metering, common area energy, EPC per dwelling unit, RdSAP methodology for UK, DPE for France, benchmarked against national dwelling stock |
| 7 | `mixed_use_development.yaml` | Mixed-Use Development | Multiple use zones with different schedules, shared central plant (CHP, heat pump, district heat), complex metering with sub-metering per zone, separate EPC per unit, common area energy allocation |
| 8 | `public_sector_building.yaml` | Public Sector (Government/Library/Museum) | Heritage/listed building constraints on external insulation, variable public access hours, display/storage environmental control (museums), server rooms, DEC mandatory for public buildings >250m2, benchmarked against CIBSE TM46 Type 6/8 |

---

## 4. Engine Specifications

### 4.1 Engine 1: Building Envelope Engine

**Purpose:** Calculate thermal performance of the building envelope including U-values, thermal bridges, air tightness, condensation risk, and window performance.

**U-Value Calculation per EN ISO 6946:**

```
U = 1 / R_total

R_total = R_si + R_1 + R_2 + ... + R_n + R_se

Where:
  R_si = internal surface resistance (m2.K/W)
       = 0.13 (horizontal heat flow - walls)
       = 0.10 (upward heat flow - roofs)
       = 0.17 (downward heat flow - floors)
  R_n = d_n / lambda_n for each construction layer
       d_n = layer thickness (m)
       lambda_n = thermal conductivity (W/m.K)
  R_se = external surface resistance (m2.K/W)
       = 0.04 (exposed to external air)
       = 0.13 (sheltered/adjacent unheated space)
```

**Typical Material Thermal Conductivities:**

| Material | Lambda (W/m.K) | Source |
|----------|---------------|--------|
| Dense concrete (2100 kg/m3) | 1.40 | EN ISO 10456 |
| Lightweight concrete (1400 kg/m3) | 0.57 | EN ISO 10456 |
| Engineering brick | 1.56 | CIBSE Guide A |
| Common brick | 0.77 | CIBSE Guide A |
| Aerated concrete block | 0.15-0.22 | EN ISO 10456 |
| Mineral wool insulation | 0.035-0.040 | Manufacturer data |
| EPS (Expanded Polystyrene) | 0.032-0.038 | EN 13163 |
| XPS (Extruded Polystyrene) | 0.028-0.034 | EN 13164 |
| PIR/PUR foam board | 0.020-0.025 | EN 13165 |
| Phenolic foam board | 0.018-0.022 | EN 13166 |
| Aerogel insulation | 0.013-0.018 | Manufacturer data |
| Timber (softwood) | 0.13 | EN ISO 10456 |
| Steel | 50.0 | EN ISO 10456 |
| Plasterboard | 0.21 | EN ISO 10456 |
| Glass (single pane) | 1.05 | EN ISO 10456 |

**Thermal Bridge Assessment per EN ISO 10211:**

| Thermal Bridge Type | Typical Psi-Value (W/m.K) | Accredited Detail Psi-Value |
|---------------------|---------------------------|----------------------------|
| Wall-floor junction (uninsulated) | 0.50-0.80 | 0.05-0.16 |
| Wall-floor junction (insulated) | 0.10-0.30 | 0.04-0.08 |
| Wall-roof junction | 0.10-0.30 | 0.04-0.08 |
| Window jamb (masonry, uninsulated) | 0.15-0.30 | 0.02-0.06 |
| Window sill | 0.04-0.15 | 0.02-0.04 |
| Window head/lintel (steel) | 0.20-0.50 | 0.03-0.08 |
| Corner (external) | 0.05-0.15 | 0.02-0.05 |
| Intermediate floor-wall junction | 0.05-0.20 | 0.02-0.07 |
| Party wall-external wall junction | 0.04-0.15 | 0.02-0.05 |
| Balcony (cantilevered concrete) | 0.60-1.20 | 0.04-0.12 (thermal break) |

**Thermal Bridge Correction:**

```
U_corrected = U_element + delta_U_tb

delta_U_tb = Sum(psi_j * L_j) / A_element + Sum(chi_k) / A_element

Where:
  psi_j = linear thermal transmittance of bridge j (W/m.K)
  L_j = length of bridge j (m)
  chi_k = point thermal transmittance of bridge k (W/K)
  A_element = area of building element (m2)

Default thermal bridging allowance (if detailed calculation not available):
  y-value = 0.15 W/m2.K (default, non-accredited)
  y-value = 0.08 W/m2.K (accredited construction details)
  y-value = 0.04 W/m2.K (thermal-bridge-free construction)
```

**Air Tightness per EN ISO 9972:**

| Metric | Formula | Typical Values |
|--------|---------|----------------|
| n50 (air changes/hr at 50 Pa) | V_50 / V_building | Building regs: 3-10 ACH; Passive House: <= 0.6 ACH |
| q50 (m3/hr/m2 at 50 Pa) | V_50 / A_envelope | Building regs: 3-7 m3/hr/m2; best practice: <1 m3/hr/m2 |
| w50 (m3/hr/m at 50 Pa) | V_50 / L_joints | Used for joint length analysis |

**Infiltration Rate from Air Tightness:**

```
n_inf = n50 / N_factor

Where:
  N_factor = 20 (sheltered site, single-sided exposure)
  N_factor = 15 (semi-exposed site)
  N_factor = 10 (exposed site, multi-directional exposure)

Infiltration heat loss:
Q_inf = 0.33 * n_inf * V_building * delta_T (W)
Annual_infiltration_loss = Q_inf * degree_hours / 1000 (kWh)
```

**Window Performance per EN ISO 10077-1:**

```
Uw = (Ag * Ug + Af * Uf + Lg * psi_g) / (Ag + Af)

Where:
  Uw = window thermal transmittance (W/m2.K)
  Ag = glazing area (m2)
  Ug = glazing U-value (W/m2.K)
  Af = frame area (m2)
  Uf = frame U-value (W/m2.K)
  Lg = glazing perimeter (m)
  psi_g = spacer bar psi-value (W/m.K)
```

**Typical Window Configurations:**

| Glazing Type | Ug (W/m2.K) | g-value | Light Transmittance |
|-------------|-------------|---------|---------------------|
| Single glazing (4mm) | 5.7 | 0.85 | 0.90 |
| Double glazing (air filled) | 2.7-3.0 | 0.75 | 0.80 |
| Double glazing (argon filled, low-e) | 1.1-1.4 | 0.50-0.65 | 0.70-0.78 |
| Triple glazing (argon filled, 2x low-e) | 0.5-0.7 | 0.35-0.55 | 0.55-0.70 |
| Triple glazing (krypton filled, 2x low-e) | 0.4-0.6 | 0.35-0.50 | 0.55-0.65 |
| Vacuum glazing | 0.4-0.7 | 0.50-0.65 | 0.65-0.80 |

**Condensation Risk Assessment (Glaser Method per EN ISO 13788):**

```
For each interface between construction layers:
  1. Calculate temperature at each interface: T_i = T_si + (T_se - T_si) * (R_si_to_i / R_total)
  2. Calculate saturated vapour pressure: p_sat(T) = 610.5 * exp(17.269 * T / (237.3 + T)) (Pa, for T >= 0)
  3. Calculate actual vapour pressure at each interface using vapour resistance
  4. If actual vapour pressure > saturated vapour pressure at any interface, condensation occurs

Vapour resistance: R_v = d / delta_v
Where:
  d = layer thickness (m)
  delta_v = vapour permeability (kg/m.s.Pa)
  mu = vapour resistance factor (dimensionless)
  delta_v = delta_air / mu
  delta_air = 2.0 x 10^-10 kg/m.s.Pa (at 25C)
```

**Key Models:**
- `EnvelopeInput` - Building geometry, construction layers (material, thickness, lambda), window schedule (type, area, orientation), thermal bridge details, air tightness test data
- `EnvelopeResult` - U-values per element, area-weighted mean U-value, thermal bridge correction, air tightness metrics, condensation risk assessment, fabric heat loss coefficient
- `UValueCalculation` - Layer-by-layer R-value build-up, total R-value, U-value, thermal bridge adjustment
- `WindowPerformance` - Uw, g-value, light transmittance, frame fraction, spacer psi-value
- `CondensationAssessment` - Month-by-month Glaser analysis, condensation risk status, interstitial moisture accumulation

**Edge Cases:**
- Unknown construction layers -> Use age-of-building defaults from national typology databases (TABULA/EPISCOPE)
- Cavity wall (unclear insulation status) -> Offer both uninsulated (R_cavity = 0.18) and insulated (R_cavity = variable) scenarios
- Mixed construction types in single element -> Area-weighted U-value calculation
- Green roof / living wall -> Specific thermal and moisture models with seasonal vegetation effect
- Historical / listed building -> Heritage constraints flag, internal insulation only option

**Non-Functional Requirements:**
- U-value calculation per building element: <1 second
- Full envelope assessment (50 elements, 100 windows): <30 seconds
- Condensation risk assessment (12-month Glaser): <5 seconds per element
- Reproducibility: bit-perfect (same input produces same output, SHA-256 verified)

### 4.2 Engine 2: EPC Rating Engine

**Purpose:** Generate Energy Performance Certificate ratings A-G per EPBD methodology with support for multiple national calculation methods.

**EPC Rating Framework per EPBD:**

| Rating | Primary Energy (kWh/m2/yr) | Description |
|--------|---------------------------|-------------|
| A | <50 | Nearly zero-energy / zero-emission building |
| B | 50-100 | Very energy efficient |
| C | 100-150 | Energy efficient |
| D | 150-200 | Average |
| E | 200-250 | Below average |
| F | 250-300 | Poor |
| G | >300 | Very poor (worst performing) |

Note: Exact thresholds vary by Member State and building type. Above values are indicative for non-residential buildings.

**Primary Energy Calculation per EN 15603:**

```
EP = Sum(f_p,del,i * E_del,i) - Sum(f_p,exp,j * E_exp,j)

Where:
  EP = primary energy (kWh/m2/yr)
  f_p,del,i = primary energy factor for delivered energy carrier i
  E_del,i = delivered energy for carrier i (kWh/m2/yr)
  f_p,exp,j = primary energy factor for exported energy carrier j
  E_exp,j = exported energy for carrier j (kWh/m2/yr)
```

**Primary Energy Factors (EU Typical Values):**

| Energy Carrier | Non-Renewable PE Factor | Total PE Factor | CO2 Factor (kgCO2/kWh) |
|---------------|------------------------|-----------------|------------------------|
| Grid electricity | 1.5-2.5 (varies by MS) | 1.8-2.8 | 0.20-0.60 (varies by grid mix) |
| Natural gas | 1.1 | 1.1 | 0.202 |
| Heating oil | 1.1 | 1.1 | 0.265 |
| LPG | 1.1 | 1.1 | 0.227 |
| District heating (CHP) | 0.7-1.0 | 0.7-1.3 | 0.10-0.25 |
| District cooling | 0.8-1.2 | 0.9-1.4 | 0.05-0.20 |
| Biomass (wood pellets) | 0.2 | 1.2 | 0.019 (biogenic excluded) |
| Solar thermal | 0.0 | 1.0 | 0.0 |
| Solar PV (on-site) | 0.0 | 1.0 | 0.0 |
| Exported electricity (to grid) | 1.5-2.5 | 1.8-2.8 | Grid displacement factor |

**Energy Needs Calculation per EN ISO 52016-1 (Monthly Method):**

```
Heating energy need (monthly):
Q_H,nd = Q_H,tr + Q_H,ve - eta_H,gn * (Q_int + Q_sol)

Where:
  Q_H,tr = transmission heat loss = Sum(H_D * (T_int - T_ext) * t_month)
  Q_H,ve = ventilation heat loss = Sum(H_V * (T_int - T_ext) * t_month)
  H_D = fabric heat loss coefficient = Sum(U_i * A_i) + Sum(psi_j * L_j)
  H_V = ventilation heat loss coefficient = 0.33 * n * V
  eta_H,gn = utilization factor for heat gains (0.7-0.99)
  Q_int = internal heat gains (occupancy, lighting, equipment)
  Q_sol = solar heat gains through glazing = Sum(g_eff * A_w * I_sol)

Cooling energy need (monthly):
Q_C,nd = Q_int + Q_sol - eta_C,ls * (Q_C,tr + Q_C,ve)

Where:
  eta_C,ls = utilization factor for heat losses (0.7-0.99)
```

**Delivered Energy Calculation:**

```
Delivered energy for heating:
E_del,H = Q_H,nd / eta_H,gen / eta_H,dist / eta_H,em / eta_H,ctrl

Where:
  eta_H,gen = generation efficiency (boiler: 0.85-0.95, heat pump: COP 2.5-5.0)
  eta_H,dist = distribution efficiency (0.85-0.98)
  eta_H,em = emission efficiency (radiators: 0.93-0.97, UFH: 0.95-0.99)
  eta_H,ctrl = control efficiency (0.88-0.98 per EN 15232)

Similarly for cooling, DHW, lighting, ventilation fans, pumps.
```

**National Methodology Variants:**

| Country | Methodology | Rating Metric | Key Differences |
|---------|-------------|---------------|-----------------|
| UK (SAP 10.2) | BRE SAP | CO2 emissions (kgCO2/m2/yr) -> EER/TER comparison | SAP rating 0-100+, EPC band A-G, carbon-based primary metric |
| Germany (GEG) | DIN V 18599 | Primary energy demand (kWh/m2/yr) | Reference building method, Energieausweis A+/A to H |
| France (DPE 2021) | 3CL-DPE | Dual: energy (kWh/m2/yr) + GHG (kgCO2/m2/yr) | Worst-of-two determines final rating, 5-usage scope |
| Italy (APE) | UNI/TS 11300 | Non-renewable EP index (kWh/m2/yr) | 10 classes A4 to G, renewable energy contribution |
| Netherlands (NTA 8800) | BENG indicators | Energy demand + primary energy + renewable share | Three BENG criteria must all pass |
| Spain (CALENER) | CTE HE methodology | CO2 emissions index + primary energy | Alphabetic rating with reference building comparison |

**Key Models:**
- `EPCInput` - Building geometry (area, volume, envelope), construction data, HVAC systems, lighting, DHW, renewables, occupancy, national jurisdiction
- `EPCResult` - EPC rating (A-G), primary energy (kWh/m2/yr), CO2 emissions (kgCO2/m2/yr), delivered energy by carrier, energy needs by end use, recommendations
- `EnergyBalance` - Monthly heating/cooling/DHW/lighting/ventilation energy needs, delivered energy, primary energy, exported energy
- `EPCRecommendations` - Ranked list of improvement measures with predicted post-retrofit rating improvement
- `NationalMethodology` - Country-specific calculation parameters, rating thresholds, primary energy factors

**Edge Cases:**
- Mixed-use building -> Separate EPCs per use zone with common area allocation
- Part-heated building -> Include unheated zone as thermal buffer with adjusted boundary
- Building with CHP -> Complex fuel allocation between heat and electricity using Carnot method or heat quality method
- Listed building -> Flag heritage constraints, relaxed MEPS timeline per EPBD Article 9(1)
- Building under construction -> Design-stage EPC (asset rating) with as-designed vs. as-built comparison

**Non-Functional Requirements:**
- EPC calculation per building (monthly method): <30 seconds
- EPC calculation per building (hourly method): <5 minutes
- Batch EPC processing (100 dwellings): <15 minutes
- 100% accuracy match with national reference calculation tool (within rounding tolerance)

### 4.3 Engine 3: HVAC Assessment Engine

**Purpose:** Assess heating, ventilation, and air conditioning system efficiency including seasonal performance, part-load behaviour, and refrigerant compliance.

**Heating System Efficiency:**

| System Type | Efficiency Metric | Typical Range | Best Practice |
|-------------|------------------|---------------|---------------|
| Gas boiler (non-condensing) | AFUE | 78-85% | N/A (phase-out) |
| Gas boiler (condensing) | AFUE / Seasonal efficiency | 89-96% | >94% |
| Oil boiler | AFUE | 80-92% | >90% |
| Biomass boiler (wood pellet) | Seasonal efficiency | 80-92% | >88% |
| Air source heat pump (ASHP) | SCOP per EN 14825 | 2.5-4.5 | >3.8 (climate zone avg) |
| Ground source heat pump (GSHP) | SCOP per EN 14825 | 3.5-5.5 | >4.5 |
| Water source heat pump | SCOP per EN 14825 | 3.0-5.0 | >4.2 |
| District heating | Distribution efficiency | 85-95% | >92% |
| Direct electric | COP | 1.0 | N/A (highest cost) |
| Electric storage heater | System efficiency | 0.85-1.0 | >0.95 (high retention) |

**Heat Pump SPF Calculation (System Performance Factor):**

```
SPF boundaries per EN 15316-4-2:

SPF1 = Compressor only
SPF2 = SPF1 + source pump (ground loop pump, defrost)
SPF3 = SPF2 + distribution pump
SPF4 = SPF3 + supplementary heater (backup boiler/immersion)

SPF4 = Q_heat_delivered / (W_compressor + W_source_pump + W_dist_pump + W_supplementary)

Bivalent operation:
  - Bivalent temperature T_biv (typically -2C to +5C depending on climate)
  - Below T_biv: heat pump + supplementary heater operate simultaneously
  - Heat pump contribution fraction = Q_hp / Q_total
  - SPF degradation at low temperatures: COP(T_source) = COP_nom * (T_source + 273) / (T_nom + 273) * F_correction
```

**Cooling System Efficiency per EN 14511/14825:**

| System Type | Efficiency Metric | Typical Range | Best Practice |
|-------------|------------------|---------------|---------------|
| Split system (inverter) | SEER | 4.0-8.0 | >6.5 |
| Split system (fixed speed) | SEER | 3.0-5.0 | >4.5 |
| VRF/VRV system | SEER | 4.5-8.5 | >7.0 |
| Chiller (air-cooled, scroll) | SEER | 3.0-5.0 | >4.5 |
| Chiller (water-cooled, centrifugal) | SEER | 5.0-9.0 | >7.5 |
| District cooling | Distribution efficiency | 85-95% | >90% |
| Absorption chiller (single effect) | COP | 0.6-0.8 | >0.75 |
| Absorption chiller (double effect) | COP | 1.0-1.4 | >1.2 |
| Evaporative cooling | Indirect effectiveness | 60-80% | >75% |
| Free cooling (economizer) | Hours of free cooling | 2000-5000 hrs/yr | Maximize by climate |

**SEER Calculation per EN 14825:**

```
SEER = Q_C / Q_CE

Where:
  Q_C = reference annual cooling demand (kWh)
  Q_CE = reference annual energy consumption for cooling (kWh)

Q_CE = Sum(h_j * P_DC(T_j) / EER(T_j)) for j = A,B,C,D operating points

Part-load ratios:
  Part load A: 100% at T_outdoor = 35C
  Part load B: 74% at T_outdoor = 30C
  Part load C: 47% at T_outdoor = 25C
  Part load D: 21% at T_outdoor = 20C

Bin hours per EN 14825 climate zones (average, warmer, colder)
```

**Ventilation Heat Recovery:**

| System Type | Effectiveness | Specific Fan Power (SFP) | Best Practice |
|-------------|--------------|--------------------------|---------------|
| Plate heat exchanger (cross-flow) | 55-75% | 0.5-1.0 W/(l/s) | >70%, SFP <0.8 |
| Plate heat exchanger (counter-flow) | 75-90% | 0.6-1.2 W/(l/s) | >85%, SFP <0.9 |
| Rotary wheel (total energy) | 70-85% (sensible), 50-70% (latent) | 0.6-1.0 W/(l/s) | >80%, SFP <0.8 |
| Run-around coil | 45-65% | 0.8-1.5 W/(l/s) | >60%, SFP <1.0 |
| Heat pipe | 50-70% | 0.5-0.8 W/(l/s) | >65%, SFP <0.7 |
| None (natural ventilation) | 0% | 0 | N/A |

**Building Automation and Controls Efficiency per EN 15232:**

| BACS Class | Description | Heating Factor | Cooling Factor | Lighting Factor |
|------------|-------------|---------------|----------------|-----------------|
| A | High energy performance BACS and TBM | 0.70-0.81 | 0.70-0.80 | 0.72-0.82 |
| B | Advanced BACS and some TBM | 0.80-0.88 | 0.80-0.88 | 0.84-0.90 |
| C | Standard BACS (reference) | 1.00 | 1.00 | 1.00 |
| D | No BACS / no energy efficiency functionality | 1.10-1.51 | 1.10-1.20 | 1.06-1.10 |

**F-Gas Regulation Refrigerant Tracking:**

| Refrigerant | GWP (AR5) | Phase-Down Status | Leak Check Frequency |
|-------------|-----------|-------------------|---------------------|
| R-410A | 2088 | Phase-down in progress | >= 5 tCO2e: 12 months; >= 50 tCO2e: 6 months; >= 500 tCO2e: 3 months + leak detection |
| R-32 | 675 | Preferred low-GWP alternative | As above per tCO2e threshold |
| R-134a | 1430 | Phase-down | As above |
| R-290 (propane) | 3 | Natural refrigerant, preferred | Exempt from F-gas leak checks |
| R-744 (CO2) | 1 | Natural refrigerant, preferred | Exempt from F-gas leak checks |
| R-717 (ammonia) | 0 | Natural refrigerant | Not F-gas regulated |
| R-454B | 466 | Low-GWP replacement for R-410A | As above per tCO2e threshold |
| R-1234yf | <1 | HFO, preferred alternative | Exempt from F-gas leak checks |

**Key Models:**
- `HVACInput` - Heating system details (type, capacity, age, fuel), cooling system details, ventilation system (type, SFP, heat recovery), controls classification, refrigerant inventory
- `HVACResult` - Heating seasonal efficiency, cooling SEER/EER, heat pump SPF1-4, ventilation heat recovery effectiveness, BACS classification, F-gas compliance status
- `HeatingAssessment` - System type, AFUE/SCOP, distribution efficiency, emission efficiency, controls factor, bivalent operation
- `CoolingAssessment` - System type, SEER, part-load ratios, free cooling potential, refrigerant status
- `VentilationAssessment` - Ventilation rate, SFP, heat recovery effectiveness, duct leakage, demand control potential

**Edge Cases:**
- Hybrid heating (heat pump + gas boiler) -> Bivalent calculation with switchover temperature and run-fraction
- VRF with simultaneous heating and cooling -> Heat recovery fraction calculation
- Natural ventilation building -> Ventilation heat loss via infiltration and window opening; no mechanical system to assess
- District heating with unknown generation efficiency -> Use national default or district-specific data if available
- Heat pump in cooling mode -> Reverse-cycle COP calculation
- Multiple heating zones with different systems -> Zone-weighted system efficiency

**Non-Functional Requirements:**
- HVAC assessment per building: <15 seconds
- Heat pump SPF calculation: <5 seconds
- F-gas compliance check per system: <2 seconds

### 4.4 Engine 4: Domestic Hot Water Engine

**Purpose:** Calculate DHW energy demand and system efficiency per EN 15316-3.

**DHW Demand by Building Type per EN 15316-3:**

| Building Type | DHW Demand (litres/day) | Demand Basis | Design Temperature |
|---------------|------------------------|--------------|-------------------|
| Residential (per person) | 36-48 litres at 60C | EN 15316-3 Table B.1 | 60C delivery, 10C cold mains |
| Office (per person per day) | 5-10 litres at 60C | CIBSE Guide G | 60C delivery |
| Hotel (per room per day) | 100-200 litres at 60C | CIBSE Guide G | 60C delivery |
| Hospital (per bed per day) | 100-150 litres at 60C | CIBSE Guide G | 60C delivery |
| School (per pupil per day) | 5-10 litres at 60C | CIBSE Guide G | 60C delivery |
| Sports centre (per user) | 30-40 litres at 60C | CIBSE Guide G | 60C delivery |
| Restaurant (per cover) | 8-15 litres at 60C | CIBSE Guide G | 60C delivery |
| Retail | Negligible | Typically staff only | 60C delivery |

**DHW Energy Demand:**

```
Q_DHW = V_DHW * rho * Cp * (T_hot - T_cold) / 3600

Where:
  Q_DHW = daily DHW energy demand (kWh/day)
  V_DHW = daily DHW volume demand (litres/day)
  rho = water density = 1.0 kg/litre
  Cp = specific heat of water = 4.186 kJ/kg.K
  T_hot = delivery temperature (typically 60C)
  T_cold = cold water inlet temperature (typically 5-15C, varies seasonally)
```

**DHW System Efficiency:**

| Component | Efficiency Factor | Typical Range |
|-----------|------------------|---------------|
| Generation (gas boiler) | eta_gen | 0.80-0.95 |
| Generation (heat pump, dedicated DHW) | COP | 2.0-4.0 (varies with source/sink temp) |
| Generation (direct electric immersion) | eta_gen | 0.95-1.0 |
| Storage (cylinder standing loss) | kWh/day loss | 0.5-3.0 kWh/day (depends on insulation, volume) |
| Primary circulation (pumped) | eta_dist_primary | 0.85-0.95 |
| Secondary circulation (constant) | eta_dist_secondary | 0.80-0.95 (higher loss from continuous pumping) |
| Distribution (dead-leg) | eta_dist_deadleg | 0.90-0.98 (loss = volume in dead-legs x temperature x events) |

**Solar Thermal Contribution (f-chart Method):**

```
Solar fraction (f) = 1.029*Y - 0.065*X - 0.245*Y^2 + 0.0018*X^2 + 0.0215*Y^3

Where:
  X = (Ac * FR * UL * (T_ref - T_amb_avg) * delta_t) / Q_DHW_monthly
  Y = (Ac * FR * tau_alpha * HT * N) / Q_DHW_monthly

  Ac = collector area (m2)
  FR = collector heat removal factor
  UL = collector heat loss coefficient (W/m2.K)
  tau_alpha = transmittance-absorptance product
  HT = monthly average daily radiation on collector plane (kWh/m2/day)
  N = number of days in month
  T_ref = reference temperature (100C for liquid systems)
  T_amb_avg = monthly average ambient temperature (C)
  Q_DHW_monthly = monthly DHW energy demand (kWh)

Annual solar fraction = Sum(f_monthly * Q_DHW_monthly) / Sum(Q_DHW_monthly)
```

**Legionella Compliance:**

| Requirement | Value | Standard |
|-------------|-------|---------|
| Storage temperature | >= 60C | HSG274 Part 2, EN 806-2 |
| Distribution return temperature | >= 50C (ideally >= 55C) | HSG274 Part 2 |
| Weekly pasteurization (if stored < 60C) | 70C for 1 minute at all outlets | CIBSE TM13 |
| Cold water temperature | <= 20C (within 2 minutes of turning on) | HSG274 Part 2 |
| Pasteurization energy penalty | 5-15% of DHW energy | Calculation based on cycle frequency |

**Key Models:**
- `DHWInput` - Building type, occupancy, DHW system type, storage volume, insulation, distribution layout, solar thermal collector data (if present)
- `DHWResult` - Annual DHW demand (kWh), system efficiency, generation energy, storage losses, distribution losses, solar thermal contribution, legionella compliance status
- `SolarThermalAssessment` - Collector area, orientation, tilt, monthly solar fraction, annual solar fraction, savings
- `DHWSystemComparison` - Side-by-side comparison of DHW system options (gas boiler, heat pump, solar thermal, hybrid)

### 4.5 Engine 5: Lighting Assessment Engine

**Purpose:** Evaluate lighting energy performance per EN 15193 and EN 12464.

**LENI Calculation per EN 15193-1:**

```
LENI = (W_L + W_P) / A (kWh/m2/yr)

Where:
  W_L = annual lighting energy (kWh)
  W_P = annual parasitic energy for emergency/controls (kWh)
  A = useful floor area (m2)

W_L = Sum( P_n,i * F_C,i * F_O,i * t_D,i + P_n,i * F_C,i * F_A,i * t_N,i ) / 1000

Where:
  P_n,i = installed lighting power in zone i (W)
  F_C,i = constant illuminance factor (0.90-1.00)
  F_O,i = occupancy dependency factor (0.60-1.00)
  F_A,i = absence factor (daylight hours when unoccupied, 0.10-1.00)
  t_D,i = daylight hours per year in zone i
  t_N,i = non-daylight operating hours per year in zone i
```

**Lighting Power Density Benchmarks per EN 12464-1:**

| Space Type | Required Illuminance (lux) | LPD Target (W/m2) | LPD Best Practice (W/m2) |
|------------|--------------------------|-------------------|--------------------------|
| Open-plan office | 500 | 10-12 | 6-8 (LED) |
| Cellular office | 500 | 10-12 | 6-8 (LED) |
| Meeting room | 500 | 10-12 | 6-8 (LED) |
| Circulation / corridor | 100 | 4-6 | 2-3 (LED) |
| Toilets | 200 | 5-8 | 3-4 (LED) |
| Reception / lobby | 300 | 8-12 | 5-7 (LED) |
| Retail (general) | 300 | 12-20 | 8-12 (LED) |
| Retail (food display) | 500 | 15-25 | 10-15 (LED) |
| Classroom | 300-500 | 10-14 | 6-8 (LED) |
| Hospital ward | 300 | 8-12 | 5-7 (LED) |
| Operating theatre | 1000 | 30-50 | 20-30 (LED) |
| Restaurant | 200-300 | 8-12 | 5-8 (LED) |
| Car park | 75 | 2-4 | 1-2 (LED) |
| Warehouse | 200 | 6-10 | 3-5 (LED, high-bay) |

**Lighting Control Credits:**

| Control Type | Occupancy Factor (F_O) Reduction | Typical Savings |
|-------------|--------------------------------|-----------------|
| Manual on/off only | F_O = 1.00 | Baseline |
| Occupancy/absence detection (auto on/off) | F_O = 0.75-0.90 | 10-25% |
| Absence detection only (manual on, auto off) | F_O = 0.65-0.80 | 20-35% |
| Daylight dimming (continuous) | Reduces t_D component | 20-40% (perimeter zones) |
| Daylight dimming + absence | Combined effect | 35-55% (perimeter zones) |
| Time scheduling | Reduces total operating hours | 10-20% |
| Personal dimming/switching | F_O = 0.85-0.95 | 5-15% |
| Constant illuminance (maintenance factor) | F_C = 0.90 | 5-10% |

**Daylight Factor Assessment per EN 17037:**

```
Daylight Factor (DF) = (E_internal / E_external) * 100 (%)

Target DF per EN 17037 recommendations:
  Minimum target illuminance (300 lux): DF >= 2% over >= 50% of reference plane
  Minimum target illuminance (100 lux): DF >= 0.7% over >= 95% of reference plane

Simplified DF estimation:
DF = (Ag * tau * theta) / (A_total * (1 - R_avg^2)) * 100

Where:
  Ag = glazing area (m2)
  tau = glazing light transmittance (0.55-0.80)
  theta = angle of visible sky from point on work plane (0-90 degrees)
  A_total = total room surface area (m2)
  R_avg = area-weighted average reflectance (walls, ceiling, floor)
```

**Key Models:**
- `LightingInput` - Space inventory (type, area, required lux), luminaire schedule (type, power, efficacy), controls inventory, glazing data for daylight
- `LightingResult` - LENI per zone and building total, LPD per zone, daylight factor, controls credit assessment, upgrade recommendations with savings
- `LEDRetrofitAnalysis` - Per-space LED replacement specification, power reduction, savings (kWh/yr, EUR/yr), payback
- `DaylightAssessment` - Daylight factor per zone, autonomy percentage, glare risk, control potential

### 4.6 Engine 6: Renewable Integration Engine

**Purpose:** Assess on-site and near-site renewable energy potential for buildings.

**Solar PV Yield Estimation:**

```
Annual PV yield (kWh) = P_peak * PSH * PR * (1 - annual_degradation * age)

Where:
  P_peak = installed peak power (kWp) = module_area * module_efficiency * 1000
  PSH = Peak Sun Hours (kWh/m2/day equivalent) by location, orientation, tilt
  PR = Performance Ratio (0.75-0.85 typical)
  PR = eta_inverter * eta_cable * eta_soiling * eta_mismatch * eta_temperature

Temperature correction:
  eta_temperature = 1 + gamma * (T_cell - T_STC)
  T_cell = T_ambient + (NOCT - 20) * G / 800
  gamma = temperature coefficient (-0.3 to -0.5 %/C for silicon)
  T_STC = 25C (Standard Test Conditions)

Shading loss:
  Shading factor = f(horizon profile, adjacent buildings, self-shading for arrays)
  Row spacing for flat roof arrays: D >= H / tan(solar_altitude_at_winter_solstice)
```

**Typical PV Yields by European City:**

| City | Latitude | Annual Irradiation (kWh/m2, optimal tilt) | Yield (kWh/kWp/yr, south-facing optimal) |
|------|----------|------------------------------------------|------------------------------------------|
| London | 51.5N | 1,050-1,150 | 850-950 |
| Paris | 48.9N | 1,150-1,250 | 950-1,050 |
| Berlin | 52.5N | 1,000-1,100 | 850-950 |
| Amsterdam | 52.4N | 1,000-1,100 | 850-950 |
| Madrid | 40.4N | 1,600-1,800 | 1,350-1,550 |
| Rome | 41.9N | 1,450-1,650 | 1,250-1,400 |
| Stockholm | 59.3N | 900-1,000 | 800-900 |
| Dublin | 53.3N | 950-1,050 | 800-900 |
| Athens | 37.9N | 1,650-1,850 | 1,400-1,600 |
| Lisbon | 38.7N | 1,600-1,800 | 1,400-1,550 |

**Solar Thermal Collector Efficiency:**

```
eta_collector = eta_0 - a1 * (T_m - T_a) / G - a2 * (T_m - T_a)^2 / G

Where:
  eta_0 = optical efficiency (zero-loss) (0.70-0.85 for flat plate, 0.60-0.75 for evacuated tube)
  a1 = first-order heat loss coefficient (W/m2.K) (3.5-4.5 for flat plate, 1.0-2.0 for evacuated tube)
  a2 = second-order heat loss coefficient (W/m2.K2) (0.01-0.02 for flat plate, 0.005-0.01 for evacuated tube)
  T_m = mean collector temperature (C)
  T_a = ambient temperature (C)
  G = solar irradiance (W/m2)
```

**Renewable Fraction Calculation per EPBD:**

```
Renewable_fraction = E_renewable / E_total_delivered

Where:
  E_renewable = E_solar_thermal + E_PV_on_site + E_heat_pump_renewable + E_biomass + E_wind
  E_heat_pump_renewable = Q_hp_delivered * (1 - 1/SPF)  [only if SPF > 2.5 per RED III]
  E_total_delivered = total delivered energy to the building
```

**Key Models:**
- `RenewableInput` - Roof area/orientation/tilt/shading, solar resource data, electricity consumption profile, DHW demand, building location and climate zone
- `RenewableResult` - PV potential (kWp, kWh/yr, self-consumption %, export %), solar thermal fraction, renewable energy fraction, financial analysis (NPV, payback, LCOE)
- `PVAssessment` - System sizing, annual yield, monthly profile, self-consumption vs. export, battery storage assessment
- `SolarThermalAssessment` - Collector sizing, annual solar fraction, monthly performance, storage requirements
- `RenewableFinancials` - CapEx, annual savings, feed-in tariff/net metering revenue, NPV, payback, LCOE comparison

### 4.7 Engine 7: Building Benchmark Engine

**Purpose:** Benchmark building energy performance against peers using multiple benchmarking frameworks.

**EUI (Energy Use Intensity) Benchmarks by Building Type:**

| Building Type | CIBSE TM46 Category | Typical EUI (kWh/m2/yr) | Good Practice EUI | Best Practice EUI |
|---------------|---------------------|------------------------|--------------------|-------------------|
| General office (naturally ventilated) | Type 1 | 120-180 | 95-120 | <95 |
| General office (air-conditioned) | Type 2 | 200-300 | 150-200 | <150 |
| Retail (general) | Type 5 | 200-350 | 165-200 | <165 |
| Retail (food) | Type 6 | 400-700 | 370-400 | <370 |
| Hotel | Type 7 | 250-400 | 200-250 | <200 |
| Hospital (general) | Type 3 | 350-500 | 300-350 | <300 |
| Primary school | Type 9 | 130-180 | 110-130 | <110 |
| Secondary school | Type 10 | 140-200 | 120-140 | <120 |
| University (teaching) | Type 11 | 150-250 | 130-150 | <130 |
| Residential (flat) | N/A | 100-200 | 80-120 | <80 |
| Restaurant | Type 14 | 350-600 | 300-350 | <300 |
| Warehouse (unheated) | Type 24 | 30-60 | 20-30 | <20 |
| Warehouse (heated) | Type 24 | 80-150 | 60-80 | <60 |
| Leisure centre (with pool) | Type 18 | 400-700 | 350-400 | <350 |

**Weather-Normalized EUI:**

```
EUI_normalized = EUI_actual * (HDD_reference / HDD_actual) * heating_fraction
                 + EUI_actual * (CDD_reference / CDD_actual) * cooling_fraction
                 + EUI_actual * baseload_fraction

Where:
  HDD_reference = long-term average heating degree days
  HDD_actual = actual year heating degree days
  CDD_reference = long-term average cooling degree days
  CDD_actual = actual year cooling degree days
  heating_fraction + cooling_fraction + baseload_fraction = 1.0
```

**Energy Star Score Estimation:**

```
Energy Star Score (1-100) is based on source EUI percentile ranking within building type.

Source EUI = Site EUI * Source-Site Ratio

Source-Site Ratios (US EPA):
  Electricity: 2.80
  Natural gas: 1.05
  District steam: 1.20
  District hot water: 1.20
  District chilled water: 0.91

Score = f(source EUI, building type, climate zone, operating hours, occupancy, etc.)
Score >= 75 qualifies for Energy Star certification
```

**CRREM Pathway Compliance Check:**

```
CRREM pathway provides annual carbon intensity targets (kgCO2/m2/yr) for:
  - Building type (60+ types: office, retail, hotel, residential, etc.)
  - Country (40+ countries)
  - Scenario (1.5C aligned, 2.0C aligned)

Stranding year = first year where building carbon intensity > CRREM pathway target

Building carbon intensity:
  CI = Sum(E_del,i * EF_i) / A_building (kgCO2/m2/yr)

Where:
  E_del,i = delivered energy for carrier i (kWh/m2/yr)
  EF_i = emission factor for carrier i (kgCO2/kWh) -- projected forward using grid decarbonization scenarios
  A_building = gross internal area (m2)

Grid decarbonization projection reduces EF_electricity over time per national pathway.
Building-level improvements (retrofit) reduce E_del.
```

**DEC Operational Rating:**

```
DEC_OR = (Actual_energy / Benchmark_energy) * 100

Where:
  Actual_energy = metered energy consumption (kWh/m2/yr), weather-corrected
  Benchmark_energy = CIBSE TM46 typical benchmark for building type (kWh/m2/yr)

DEC_OR = 100 means building performs at typical benchmark
DEC_OR < 100 means better than benchmark
DEC_OR > 100 means worse than benchmark

DEC Band:
  A: OR <= 25
  B: OR 26-50
  C: OR 51-75
  D: OR 76-100
  E: OR 101-125
  F: OR 126-150
  G: OR > 150
```

**Key Models:**
- `BenchmarkInput` - Building type, location, floor area, annual energy consumption by carrier, metering data, occupancy, operating hours
- `BenchmarkResult` - EUI (site and source), weather-normalized EUI, Energy Star score estimate, CRREM pathway status, DEC operational rating, peer percentile, trend analysis
- `CRREMAssessment` - Current carbon intensity, pathway target, gap-to-pathway, stranding year (BAU), stranding year (post-retrofit scenario), pathway scenarios (1.5C, 2.0C)
- `PeerComparison` - Percentile ranking within building type and climate zone, gap-to-median, gap-to-best-practice

### 4.8 Engine 8: Retrofit Analysis Engine

**Purpose:** Analyze building retrofit opportunities with interaction modelling and staged roadmap generation.

**Retrofit Measure Library (60+ Measures):**

**Envelope Measures (15):**

| # | Measure | Typical Savings | Typical Cost (EUR/m2) | Typical Payback |
|---|---------|----------------|----------------------|-----------------|
| 1 | External wall insulation (EWI) 100mm | 20-35% heating | 80-150 | 10-20 years |
| 2 | Internal wall insulation (IWI) 60mm | 15-25% heating | 40-80 | 8-15 years |
| 3 | Cavity wall insulation | 15-25% heating | 5-15 | 2-5 years |
| 4 | Loft/roof insulation (300mm mineral wool) | 10-20% heating | 10-25 | 2-5 years |
| 5 | Flat roof insulation (over-roof) | 10-20% heating | 60-120 | 8-15 years |
| 6 | Floor insulation (suspended timber) | 5-15% heating | 20-50 | 5-10 years |
| 7 | Floor insulation (solid ground) | 5-10% heating | 40-80 | 10-20 years |
| 8 | Double to triple glazing replacement | 10-15% heating | 300-600 per m2 glazing | 15-30 years |
| 9 | Secondary glazing | 5-10% heating | 50-150 per m2 glazing | 5-10 years |
| 10 | Draught-proofing (doors, windows) | 5-10% heating | 5-20 | 1-3 years |
| 11 | Air tightness improvement | 5-15% heating | 20-50 | 3-7 years |
| 12 | Cool roof (reflective coating) | 10-20% cooling | 10-30 | 3-7 years |
| 13 | Green roof | 5-10% heating+cooling | 80-200 | 15-25 years |
| 14 | External shading (brise soleil, louvres) | 10-25% cooling | 100-300 per m2 | 10-15 years |
| 15 | Thermal bridge remediation | 5-15% heating | 30-100 per bridge | 5-10 years |

**HVAC Measures (20):**

| # | Measure | Typical Savings | Typical Cost | Typical Payback |
|---|---------|----------------|-------------|-----------------|
| 16 | Boiler replacement (non-condensing to condensing) | 10-15% gas | EUR 3,000-8,000 | 5-10 years |
| 17 | Heat pump installation (ASHP replacing gas boiler) | 40-60% heating energy | EUR 8,000-15,000 | 7-12 years |
| 18 | Heat pump installation (GSHP) | 50-65% heating energy | EUR 15,000-25,000 | 10-15 years |
| 19 | District heating connection | 20-40% carbon reduction | EUR 10,000-30,000 | 5-15 years |
| 20 | Heating controls upgrade (TRVs, zoning) | 10-20% heating | EUR 1,000-5,000 | 2-5 years |
| 21 | Weather compensation controls | 5-10% heating | EUR 500-2,000 | 1-3 years |
| 22 | MVHR installation (with 85%+ heat recovery) | 20-30% ventilation loss | EUR 5,000-12,000 | 7-15 years |
| 23 | Demand-controlled ventilation (CO2 sensors) | 15-30% ventilation energy | EUR 2,000-8,000 | 3-7 years |
| 24 | VSD on AHU fans | 20-40% fan energy | EUR 1,000-5,000 | 2-5 years |
| 25 | Chiller replacement (high SEER) | 20-40% cooling energy | EUR 15,000-50,000 | 5-10 years |
| 26 | VRF/VRV system (replacing split systems) | 15-30% cooling energy | EUR 500-800/kW | 5-10 years |
| 27 | Free cooling / economizer | 30-50% cooling energy | EUR 3,000-10,000 | 2-5 years |
| 28 | Cooling tower optimization | 10-20% cooling energy | EUR 5,000-15,000 | 3-7 years |
| 29 | Pipework insulation improvement | 5-10% distribution loss | EUR 10-30/m pipe | 2-5 years |
| 30 | Pump replacement (high efficiency) | 20-30% pump energy | EUR 500-3,000 | 3-5 years |
| 31 | BMS upgrade / optimization | 10-20% total HVAC | EUR 20,000-100,000 | 3-7 years |
| 32 | Destratification fans (tall spaces) | 10-20% heating in tall spaces | EUR 500-2,000 each | 2-4 years |
| 33 | Refrigerant replacement (low GWP) | Regulatory compliance | EUR 2,000-10,000 | N/A (compliance) |
| 34 | Night purge ventilation | 10-20% cooling | EUR 1,000-5,000 | 2-4 years |
| 35 | Radiant ceiling panels (replacing FCU) | 10-20% HVAC energy | EUR 100-200/m2 | 8-15 years |

**Lighting Measures (10):**

| # | Measure | Typical Savings | Typical Cost | Typical Payback |
|---|---------|----------------|-------------|-----------------|
| 36 | LED replacement (fluorescent to LED) | 40-60% lighting energy | EUR 30-80 per fitting | 2-4 years |
| 37 | LED replacement (halogen to LED) | 70-80% lighting energy | EUR 10-30 per lamp | 0.5-2 years |
| 38 | LED high-bay (warehouse/industrial) | 50-70% lighting energy | EUR 200-500 per fitting | 2-4 years |
| 39 | Occupancy/absence detection | 15-30% lighting energy | EUR 50-150 per sensor | 2-5 years |
| 40 | Daylight dimming (continuous) | 20-40% perimeter zone lighting | EUR 100-300 per zone | 3-7 years |
| 41 | Task lighting (reduced ambient) | 10-20% office lighting | EUR 50-150 per desk | 3-5 years |
| 42 | External lighting LED + time control | 50-70% external lighting | EUR 100-500 per fitting | 2-5 years |
| 43 | Car park lighting LED + sensing | 60-80% car park lighting | EUR 100-400 per fitting | 2-4 years |
| 44 | Emergency lighting LED conversion | 60-70% emergency lighting | EUR 50-150 per unit | 3-5 years |
| 45 | Constant illuminance (maintenance factor) | 5-10% lighting energy | EUR 500-2,000 (controls) | 2-5 years |

**Renewable and Generation Measures (8):**

| # | Measure | Typical Generation/Savings | Typical Cost | Typical Payback |
|---|---------|--------------------------|-------------|-----------------|
| 46 | Solar PV (rooftop) | 800-1,500 kWh/kWp/yr | EUR 900-1,400/kWp | 5-10 years |
| 47 | Solar PV (BIPV facade) | 400-800 kWh/kWp/yr | EUR 1,500-3,000/kWp | 10-20 years |
| 48 | Solar thermal (flat plate) | 400-700 kWh/m2/yr | EUR 300-600/m2 | 7-12 years |
| 49 | Solar thermal (evacuated tube) | 500-900 kWh/m2/yr | EUR 500-1,000/m2 | 8-15 years |
| 50 | Battery storage (behind meter) | Peak shaving, self-consumption | EUR 500-800/kWh | 7-12 years |
| 51 | CHP / micro-CHP | 80-90% overall efficiency | EUR 1,500-3,000/kW_e | 5-8 years |
| 52 | Small wind (building-mounted) | 500-2,000 kWh/yr per unit | EUR 3,000-8,000/kW | 10-20 years |
| 53 | Heat network connection (5GDHC) | Low-carbon heat supply | EUR 15,000-30,000 | 8-15 years |

**Demand Response and Controls Measures (7):**

| # | Measure | Typical Savings | Typical Cost | Typical Payback |
|---|---------|----------------|-------------|-----------------|
| 54 | Smart metering and sub-metering | 5-10% (behaviour change) | EUR 500-2,000/meter | 2-5 years |
| 55 | Power factor correction | 2-5% electricity | EUR 5,000-20,000 | 2-4 years |
| 56 | Voltage optimization | 5-10% electricity | EUR 10,000-30,000 | 3-7 years |
| 57 | Load shifting (thermal storage) | 10-20% peak demand cost | EUR 5,000-20,000 | 3-7 years |
| 58 | Demand response participation | Revenue from flexibility | EUR 2,000-10,000 | 1-3 years |
| 59 | BEMS analytics and fault detection | 10-15% total energy | EUR 10,000-50,000 | 2-5 years |
| 60 | Tenant engagement programme | 5-10% total energy | EUR 5,000-20,000/yr | 1-2 years |

**Measure Interaction Matrix:**

| Measure A | Measure B | Interaction Type | Effect |
|-----------|-----------|-----------------|--------|
| Wall insulation | Heating system replacement | Synergy | Reduced heating load allows smaller heat pump, improves COP |
| Wall insulation | Cooling system | Interaction | Reduced cooling load (less transmission), but may increase overheating risk |
| LED lighting | Cooling system | Synergy | Reduced internal gains reduce cooling load (10-15%) |
| LED lighting | Heating system | Interaction | Reduced internal gains increase heating demand (3-5% in heating-dominated) |
| MVHR | Heating system | Synergy | Heat recovery reduces heating demand 20-30% |
| MVHR | Air tightness | Dependency | MVHR requires air tightness <= 5 m3/hr/m2 @ 50Pa to be effective |
| Solar PV | Battery storage | Synergy | Battery increases self-consumption from 30% to 60-80% |
| Heat pump | Solar PV | Synergy | PV offsets heat pump electricity, improving overall carbon reduction |
| Glazing upgrade | Solar gain reduction | Trade-off | Better U-value but lower g-value may reduce useful solar gains |

**Financial Analysis per Measure:**

```
Simple payback = CapEx / (Annual_energy_savings * Energy_price)

NPV = -CapEx + Sum( (Annual_savings * (1 + price_escalation)^t) / (1 + discount_rate)^t, t=1..life )

IRR = discount_rate where NPV = 0

Lifecycle Cost = CapEx + Sum( OpEx_t / (1 + discount_rate)^t ) - residual_value / (1 + discount_rate)^life

Marginal Abatement Cost (MAC) = (NPV of cost) / (Lifetime CO2 savings in tonnes)
```

**nZEB Gap Analysis:**

```
nZEB gap = Current_primary_energy - nZEB_target_primary_energy

nZEB target (typical, varies by MS):
  Non-residential: <= 50-90 kWh/m2/yr primary energy (depending on building type and climate)
  Residential: <= 40-75 kWh/m2/yr primary energy

Zero-Emission Building (ZEB per EPBD recast):
  - Zero on-site fossil fuel emissions from heating/DHW
  - Maximum primary energy threshold per building type
  - Minimum renewable energy share

Gap closure pathway:
  Stage 1: Quick wins (lighting, controls, draught-proofing) -> EPC improvement E->D
  Stage 2: HVAC optimization (BMS, VSDs, heat recovery) -> EPC improvement D->C
  Stage 3: Envelope deep retrofit (insulation, glazing) -> EPC improvement C->B
  Stage 4: System replacement (heat pump, MVHR) -> EPC improvement B->A
  Stage 5: Renewables (PV, solar thermal) -> nZEB/ZEB compliance
```

**Key Models:**
- `RetrofitInput` - Current building assessment results (from engines 1-7), financial parameters, energy prices, discount rate, building constraints (listed status, tenure, access)
- `RetrofitResult` - Ranked measure list with savings, costs, payback, NPV, IRR; interaction-adjusted package savings; staged roadmap; nZEB gap analysis
- `RetrofitMeasure` - Measure ID, description, energy savings (kWh/yr), cost savings (EUR/yr), CO2 savings (kgCO2/yr), CapEx, OpEx, payback, NPV, IRR, MAC, EPC improvement
- `RetrofitPackage` - Combined measures with interaction-adjusted total savings, total cost, combined payback, predicted post-retrofit EPC rating
- `RetrofitRoadmap` - Multi-stage pathway from current to target performance with per-stage investment, savings, and EPC rating

### 4.9 Engine 9: Indoor Environment Engine

**Purpose:** Evaluate indoor environmental quality including thermal comfort, air quality, ventilation adequacy, overheating risk, and daylighting.

**Thermal Comfort -- PMV/PPD per ISO 7730:**

```
PMV = [0.303 * exp(-0.036 * M) + 0.028] * L

Where:
  M = metabolic rate (W/m2)
  L = thermal load on body (W/m2)
  L = (M - W) - 3.05*10^-3 * [5733 - 6.99*(M-W) - pa]
      - 0.42 * [(M-W) - 58.15]
      - 1.7*10^-5 * M * (5867 - pa)
      - 0.0014 * M * (34 - ta)
      - 3.96*10^-8 * fcl * [(tcl+273)^4 - (tr+273)^4]
      - fcl * hc * (tcl - ta)

  ta = air temperature (C)
  tr = mean radiant temperature (C)
  pa = partial water vapour pressure (Pa)
  va = air velocity (m/s)
  M = metabolic rate (W/m2) -- office: 1.2 met = 70 W/m2
  Icl = clothing insulation (clo) -- winter office: 1.0 clo, summer: 0.5 clo
  W = external work (typically 0 for office)

PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)

ISO 7730 categories:
  Category A: -0.2 < PMV < +0.2, PPD < 6%
  Category B: -0.5 < PMV < +0.5, PPD < 10%
  Category C: -0.7 < PMV < +0.7, PPD < 15%
```

**Adaptive Thermal Comfort per EN 16798-1:**

```
For naturally ventilated / free-running buildings:

theta_comf = 0.33 * theta_rm + 18.8

Where:
  theta_comf = comfort temperature (C)
  theta_rm = running mean outdoor temperature (C)
  theta_rm = (1 - alpha) * theta_ed-1 + alpha * theta_rm,d-1
  alpha = 0.8 (weighting constant)
  theta_ed-1 = daily mean external temperature for previous day

Acceptable range:
  Category I: theta_comf +/- 2C
  Category II: theta_comf +/- 3C
  Category III: theta_comf +/- 4C

Applicability: theta_rm between 10C and 33C
```

**Indoor Air Quality Parameters per EN 16798-1:**

| Parameter | Category I | Category II | Category III | Category IV |
|-----------|-----------|------------|-------------|-------------|
| CO2 above outdoor (ppm) | 550 | 800 | 1350 | >1350 |
| Total ventilation rate (L/s per person, office) | 10 | 7 | 4 | <4 |
| Ventilation rate per floor area (L/s per m2) | 2.0 | 1.4 | 0.8 | <0.8 |
| PM2.5 (ug/m3) | <10 | <15 | <25 | >25 |
| TVOC (ug/m3) | <200 | <300 | <500 | >500 |
| Formaldehyde (ug/m3) | <30 | <50 | <100 | >100 |

**Overheating Risk Assessment per CIBSE TM59:**

```
Criterion A (living rooms, kitchens, bedrooms):
  Hours where T_operative > T_max shall not exceed 3% of occupied hours
  T_max = adaptive comfort upper limit (Category II)

Criterion B (bedrooms, 10pm-7am):
  Hours where T_operative > 26C shall not exceed 1% of annual hours (32 hours)

Assessment period: May 1 to September 30 (UK; varies by climate)

Design weather: CIBSE DSY1 (Design Summer Year, moderately warm year)
For future climate: DSY1 with 2050s or 2080s projections (UKCP18)
```

**Key Models:**
- `IEQInput` - Zone data (dimensions, occupancy, activity level), HVAC data, glazing, ventilation rates, measured/designed air temperatures, relative humidity, air velocity
- `IEQResult` - PMV/PPD per zone, adaptive comfort compliance, IAQ category per zone, overheating risk assessment, daylighting adequacy, overall IEQ score
- `ThermalComfort` - PMV, PPD, ISO 7730 category, adaptive comfort range, hours outside comfort band
- `AirQuality` - CO2 level, PM2.5, TVOC, ventilation rate per person, EN 16798-1 category
- `OverheatingRisk` - TM59 Criterion A result (% hours above threshold), Criterion B result (hours >26C), pass/fail, mitigation recommendations

### 4.10 Engine 10: Whole Life Carbon Engine

**Purpose:** Calculate whole life carbon per EN 15978 lifecycle stages.

**EN 15978 Lifecycle Stages:**

| Stage | Module | Description | Included In |
|-------|--------|-------------|-------------|
| Product | A1 | Raw material supply | Embodied Carbon |
| Product | A2 | Transport (to factory) | Embodied Carbon |
| Product | A3 | Manufacturing | Embodied Carbon |
| Construction | A4 | Transport (to site) | Embodied Carbon |
| Construction | A5 | Construction/installation | Embodied Carbon |
| Use | B1 | Use (in-situ emissions, e.g., carbonation) | Use Stage Carbon |
| Use | B2 | Maintenance | Use Stage Carbon |
| Use | B3 | Repair | Use Stage Carbon |
| Use | B4 | Replacement | Use Stage Carbon |
| Use | B5 | Refurbishment | Use Stage Carbon |
| Use | B6 | Operational energy use | Operational Carbon |
| Use | B7 | Operational water use | Operational Carbon |
| End of Life | C1 | Deconstruction/demolition | End-of-Life Carbon |
| End of Life | C2 | Transport (to waste processing) | End-of-Life Carbon |
| End of Life | C3 | Waste processing | End-of-Life Carbon |
| End of Life | C4 | Disposal | End-of-Life Carbon |
| Beyond System | D | Reuse, recovery, recycling potential | Beyond Lifecycle |

**Embodied Carbon Calculation (A1-A3):**

```
EC_A1-A3 = Sum( m_i * ECF_i )

Where:
  EC_A1-A3 = embodied carbon for product stage (kgCO2e)
  m_i = mass of material i (kg)
  ECF_i = embodied carbon factor for material i (kgCO2e/kg) from EPD or generic database
```

**Typical Embodied Carbon Factors (A1-A3):**

| Material | ECF (kgCO2e/kg) | Source |
|----------|----------------|--------|
| Concrete (C30/37, CEM I) | 0.12-0.16 | ICE Database v3 |
| Concrete (C30/37, 50% GGBS) | 0.07-0.10 | ICE Database v3 |
| Steel (structural, UK average) | 1.20-1.55 | ICE Database v3 |
| Steel (structural, EAF recycled) | 0.45-0.75 | ICE Database v3 |
| Aluminium (primary) | 6.80-9.70 | ICE Database v3 |
| Aluminium (recycled) | 0.50-1.50 | ICE Database v3 |
| Timber (softwood, CLT) | -1.60 to +0.45 | ICE Database v3 (biogenic) |
| Glulam | -0.70 to +0.50 | ICE Database v3 |
| Brick (common) | 0.21-0.24 | ICE Database v3 |
| Glass (float, clear) | 0.85-1.20 | ICE Database v3 |
| Mineral wool insulation | 1.20-1.40 | ICE Database v3 |
| EPS insulation | 3.00-3.50 | ICE Database v3 |
| PIR insulation | 3.40-4.20 | ICE Database v3 |
| Plasterboard | 0.12-0.15 | ICE Database v3 |
| Copper (pipe) | 2.70-3.80 | ICE Database v3 |

**Operational Carbon (B6) over Building Lifetime:**

```
OC_B6 = Sum_over_years( E_del,i * EF_i(year) ) / A_building

Where:
  E_del,i = annual delivered energy for carrier i (kWh)
  EF_i(year) = emission factor for carrier i in year (kgCO2/kWh)
  Grid electricity EF decreases over time with decarbonization
  Building lifetime: typically 60 years (RIBA), 50 years (LETI)
```

**Whole Life Carbon Budgets:**

| Standard | Building Type | WLC Target (kgCO2e/m2) | Study Period |
|----------|--------------|----------------------|--------------|
| RIBA 2030 (residential) | Domestic new-build | <625 | 60 years |
| RIBA 2030 (non-residential) | Office new-build | <750 | 60 years |
| LETI 2020 (residential) | Domestic new-build | <500 | 60 years |
| LETI 2020 (non-residential) | Office new-build | <600 | 60 years |
| GLA (London Plan) | All major developments | Reporting required | 60 years |
| DGNB | Various | Climate positive by 2050 | 50 years |

**Biogenic Carbon Accounting per EN 16449:**

```
Biogenic carbon in timber = -1.83 kgCO2/kg of oven-dry wood (absorption during growth)

Reporting:
  Module A1-A3: Report biogenic carbon separately (negative if stored)
  Module C3-C4: Release of biogenic carbon at end-of-life (positive)
  Net biogenic over lifecycle: zero (carbon neutral if sustainable forestry)

Temporary storage benefit:
  If timber in building for 60+ years, carbon is temporarily sequestered
  Some frameworks give credit for temporary storage (RIBA/LETI report separately)
```

**Material Substitution Analysis:**

| Original Material | Alternative | Carbon Saving (kgCO2e/kg) | Cost Impact |
|-------------------|-------------|--------------------------|-------------|
| CEM I concrete | 50% GGBS concrete | 0.05-0.08 saved | Neutral to -5% |
| CEM I concrete | 30% PFA concrete | 0.03-0.05 saved | Neutral |
| Primary steel | EAF recycled steel | 0.50-0.80 saved | +5-15% |
| Steel frame | CLT frame | 1.50-2.50 saved | Variable |
| Concrete frame | CLT frame | 0.80-1.50 saved | Variable |
| EPS insulation | Wood fibre insulation | 2.50-3.50 saved | +20-40% |
| PIR insulation | Mineral wool insulation | 2.00-3.00 saved | -10-20% |
| Aluminium cladding | Timber cladding | 6.00-9.00 saved | -20-40% |
| Primary aluminium | Recycled aluminium | 5.50-8.00 saved | +5-10% |

**Key Models:**
- `WLCInput` - Bill of materials (quantities, materials, EPD references), building energy assessment results (B6), maintenance schedule (B2-B5), design life, end-of-life scenario
- `WLCResult` - Total WLC (kgCO2e/m2), breakdown by lifecycle stage (A1-D), embodied vs. operational split, budget comparison (RIBA/LETI), material substitution recommendations
- `EmbodiedCarbon` - Per-element and per-material A1-A3 carbon, total upfront carbon (A1-A5), replacement carbon (B4) over building life
- `OperationalCarbon` - Annual B6 carbon, lifetime B6 with grid decarbonization trajectory, B7 water-related carbon
- `MaterialSubstitution` - Per-material alternative options with carbon saving, cost impact, structural/performance implications

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Initial Building Assessment Workflow

**Purpose:** End-to-end initial building energy assessment from building registration through comprehensive performance report.

**Phase 1: Building Registration**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Create building profile | Building name, address, type, year of construction, floor area (GIA/NIA), number of floors | Building record with unique ID | <5 minutes |
| 1.2 | Define building boundary | Building footprint, conditioned/unconditioned zones, adjacent buildings, extensions | Boundary definition with zone map | <10 minutes |
| 1.3 | Register building use | Primary use type, secondary uses, operating hours, occupancy levels, special requirements | Use profile per zone | <5 minutes |
| 1.4 | Configure metering infrastructure | Main meters (electricity, gas, heat), sub-meters, BMS data sources | Meter registry with hierarchy | <10 minutes |
| 1.5 | Select regulatory jurisdiction | Country, region, national EPC methodology, applicable regulations | Regulatory configuration | <2 minutes |

**Phase 2: Data Collection**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Import/enter construction data | Floor plans, section drawings, construction details, material specifications | Construction layer data per element | <30 minutes |
| 2.2 | Import utility bills | PDF/Excel utility bills (24+ months), energy procurement contracts | Standardized energy consumption time series | <15 minutes |
| 2.3 | Import BMS data | Half-hourly/hourly data from BMS (temperature, energy, setpoints) | High-resolution building performance data | <10 minutes |
| 2.4 | Survey HVAC systems | Heating, cooling, ventilation, controls: type, age, capacity, refrigerant | HVAC system inventory | <20 minutes |
| 2.5 | Data quality assessment | All imported data | Data quality score (0-100), gap report, reconciliation | <5 minutes (auto) |

**Phase 3: Envelope Analysis**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Calculate U-values | Construction layers per wall/roof/floor element | U-value per element with EN ISO 6946 compliance | <2 minutes (auto) |
| 3.2 | Assess thermal bridges | Junction details, accredited construction details where available | Psi-values, y-value, thermal bridge heat loss | <2 minutes (auto) |
| 3.3 | Assess windows | Window schedule (type, area, orientation, frame, glazing) | Uw, g-value, light transmittance per window group | <2 minutes (auto) |
| 3.4 | Assess air tightness | Air pressure test result (n50) or age-based default | Infiltration rate, infiltration heat loss | <1 minute (auto) |
| 3.5 | Calculate fabric heat loss | All envelope data | Fabric heat loss coefficient (W/K), heat loss parameter (W/m2.K) | <1 minute (auto) |

**Phase 4: Systems Assessment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Assess heating system | Heating system inventory, fuel data | Seasonal efficiency, distribution losses, emission efficiency | <5 minutes (auto) |
| 4.2 | Assess cooling system | Cooling system inventory | SEER, part-load performance, free cooling potential | <5 minutes (auto) |
| 4.3 | Assess ventilation system | AHU inventory, ductwork, controls | SFP, heat recovery effectiveness, DCV potential | <5 minutes (auto) |
| 4.4 | Assess DHW system | DHW system inventory, demand data | DHW demand, system efficiency, solar contribution | <3 minutes (auto) |
| 4.5 | Assess lighting | Luminaire inventory, controls, daylight | LENI, LPD per zone, daylight factor, control credits | <5 minutes (auto) |

**Phase 5: Performance Report**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 5.1 | Calculate energy balance | All assessment results | Monthly/annual energy balance by end use and carrier | <2 minutes (auto) |
| 5.2 | Calculate primary energy | Delivered energy, primary energy factors | Primary energy (kWh/m2/yr), CO2 emissions (kgCO2/m2/yr) | <1 minute (auto) |
| 5.3 | Benchmark performance | EUI, primary energy, CO2 | Benchmark comparison, Energy Star estimate, CRREM pathway | <2 minutes (auto) |
| 5.4 | Identify improvements | All analysis results | Prioritized retrofit measure list with savings and financials | <5 minutes (auto) |
| 5.5 | Generate report | All results | Comprehensive building assessment report (PDF/HTML/JSON) | <5 minutes (auto) |

**Acceptance Criteria:**
- [ ] All 5 phases execute sequentially with data passing between phases
- [ ] Phase 2 data quality assessment flags gaps and offers default-value substitution
- [ ] Phase 3 U-value calculations match EN ISO 6946 reference values within 1%
- [ ] Phase 4 covers all HVAC systems representing >= 90% of building energy
- [ ] Phase 5 report includes all mandatory sections per EPBD Article 16(3)
- [ ] Total workflow duration < 4 hours for a typical commercial building (data pre-loaded equivalent < 45 minutes)
- [ ] Full SHA-256 provenance chain from input data through every calculation to final report

### 5.2 Workflow 2: EPC Generation Workflow

**Purpose:** Energy Performance Certificate generation per national methodology.

**Phase 1: Building Geometry**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Define heated/conditioned volume | Floor plans, section drawings | Heated volume, floor area, height | <10 minutes |
| 1.2 | Calculate envelope areas | Building geometry | Wall, roof, floor, window areas by orientation | <5 minutes |
| 1.3 | Identify thermal zones | Use types, orientation, heating/cooling zones | Thermal zone definitions | <10 minutes |
| 1.4 | Validate geometry | Cross-check areas, volumes, ratios | Geometry consistency check (pass/fail) | <1 minute (auto) |

**Phase 2: Fabric Calculation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Calculate element U-values | Construction details per element | U-values per EN ISO 6946 | <2 minutes (auto) |
| 2.2 | Apply thermal bridge corrections | Junction details | Corrected heat loss with thermal bridging | <1 minute (auto) |
| 2.3 | Calculate window performance | Window schedule | Uw, g-value per window type | <1 minute (auto) |
| 2.4 | Calculate ventilation heat loss | Air tightness, ventilation strategy | Ventilation heat loss coefficient | <1 minute (auto) |
| 2.5 | Calculate heating/cooling energy needs | Fabric + ventilation + gains | Monthly energy needs per EN ISO 52016-1 | <2 minutes (auto) |

**Phase 3: Systems Calculation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Calculate heating energy | Heating need, system efficiency | Delivered heating energy (kWh/yr) | <1 minute (auto) |
| 3.2 | Calculate cooling energy | Cooling need, system efficiency | Delivered cooling energy (kWh/yr) | <1 minute (auto) |
| 3.3 | Calculate DHW energy | DHW demand, system efficiency | Delivered DHW energy (kWh/yr) | <1 minute (auto) |
| 3.4 | Calculate lighting energy | LENI calculation | Lighting energy (kWh/yr) | <1 minute (auto) |
| 3.5 | Calculate auxiliary energy | Fans, pumps, controls | Auxiliary energy (kWh/yr) | <1 minute (auto) |
| 3.6 | Calculate renewable contribution | PV, solar thermal output | Renewable energy offset (kWh/yr) | <1 minute (auto) |

**Phase 4: EPC Issuance**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Calculate primary energy | All delivered energy, PE factors | Primary energy (kWh/m2/yr) | <1 minute (auto) |
| 4.2 | Calculate CO2 emissions | All delivered energy, CO2 factors | CO2 emissions (kgCO2/m2/yr) | <1 minute (auto) |
| 4.3 | Determine EPC rating | Primary energy vs. rating thresholds | EPC rating A-G (or national equivalent) | <1 minute (auto) |
| 4.4 | Generate recommendations | Performance gaps, measure library | Top 5-10 improvement recommendations | <2 minutes (auto) |
| 4.5 | Produce EPC certificate | All calculation results | EPC report (PDF, HTML, JSON) | <3 minutes (auto) |

**Acceptance Criteria:**
- [ ] EPC rating matches national reference calculation tool within 1 rating band
- [ ] Primary energy calculation reproducible to 4 significant figures
- [ ] All mandatory EPC fields populated per EPBD Annex V
- [ ] Recommendations ranked by cost-effectiveness with estimated savings
- [ ] EPC valid for 10 years (validity date auto-calculated)
- [ ] Total workflow duration < 30 minutes for straightforward commercial building

### 5.3 Workflow 3: Retrofit Planning Workflow

**Purpose:** Deep retrofit planning from current performance assessment to staged implementation roadmap.

**Phase 1: Baseline Performance**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Establish current energy baseline | Utility bills, BMS data, operational data | Current EUI, primary energy, CO2, EPC rating | <10 minutes |
| 1.2 | Identify performance gaps | Current vs. benchmark, current vs. nZEB target | Gap-to-benchmark, gap-to-nZEB, gap-to-CRREM pathway | <5 minutes (auto) |
| 1.3 | Assess building constraints | Heritage status, tenure, access, structural | Constraint register with impact on measure selection | <10 minutes |
| 1.4 | Define performance targets | Regulatory minimum, nZEB, ZEB, CRREM, certification targets | Target performance levels with timeline | <5 minutes |

**Phase 2: Measure Identification**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Screen applicable measures | Building characteristics, constraints, performance gaps | Long-list of applicable measures (from 60+ library) | <5 minutes (auto) |
| 2.2 | Calculate individual measure savings | Each measure against baseline | Per-measure energy, cost, CO2 savings with payback | <10 minutes (auto) |
| 2.3 | Apply interaction matrix | Measure-measure interactions | Interaction-adjusted savings per measure | <5 minutes (auto) |
| 2.4 | Rank measures | Financial metrics (NPV, payback), CO2 impact, EPC improvement | Prioritized measure list | <2 minutes (auto) |

**Phase 3: Package Optimization**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Define packages by investment level | Budget tiers (low/medium/high/deep), time horizon | 3-5 retrofit packages with increasing depth | <5 minutes (auto) |
| 3.2 | Optimize packages for target | Target EPC rating, CRREM, nZEB | Least-cost package to achieve each target | <5 minutes (auto) |
| 3.3 | Financial analysis per package | Package costs and savings | NPV, IRR, payback, lifecycle cost per package | <3 minutes (auto) |
| 3.4 | Sensitivity analysis | Energy price, discount rate, climate scenarios | Sensitivity ranges for key financial metrics | <3 minutes (auto) |

**Phase 4: Roadmap Generation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Define implementation stages | Measure dependencies, budget profile, operational constraints | 3-5 stage retrofit roadmap with timeline | <5 minutes |
| 4.2 | Project performance trajectory | Stage-by-stage cumulative improvement | Year-by-year EPC rating, EUI, CO2 trajectory vs. CRREM pathway | <3 minutes (auto) |
| 4.3 | Assess financing options | CapEx per stage, available grants/incentives | Financing plan (grants, green loans, ESCO, EPC) | <5 minutes |
| 4.4 | Generate retrofit report | All results | Retrofit recommendation report (PDF/HTML/JSON) | <5 minutes (auto) |

**Acceptance Criteria:**
- [ ] Interaction-adjusted savings differ from sum-of-individual by 5-20% (realistic interaction)
- [ ] At least one package achieves nZEB-equivalent performance
- [ ] CRREM pathway compliance projected year-by-year for each package
- [ ] Financial analysis includes sensitivity on energy price (+/- 30%) and discount rate (+/- 2%)
- [ ] Staged roadmap respects measure dependencies (e.g., envelope before heat pump sizing)

### 5.4 Workflow 4: Continuous Building Monitoring Workflow

**Purpose:** Ongoing building energy performance monitoring with anomaly detection and trend analysis.

**Phase 1: Real-Time Data Ingestion**

| Step | Action | Frequency |
|------|--------|-----------|
| 1.1 | Ingest utility meter data | Every 30 minutes (half-hourly settlement) |
| 1.2 | Ingest BMS data | Every 5-15 minutes |
| 1.3 | Ingest sub-meter data | Every 15 minutes |
| 1.4 | Ingest weather data | Every hour |
| 1.5 | Data validation and gap detection | On every ingestion cycle |

**Phase 2: Performance Tracking**

| Metric | Tracking Method | Target |
|--------|----------------|--------|
| Daily EUI | Actual vs. degree-day-adjusted predicted | Within 10% of predicted |
| Monthly EUI by end use | Metered vs. baseline model | Within 15% of model |
| System efficiency (COP, SEER) | Calculated from metered heat output / electrical input | Within 10% of design |
| Baseload (overnight/weekend) | Minimum demand period analysis | Reducing trend (target set per building) |
| Peak demand (kW) | Maximum demand tracking | Below agreed supply capacity |
| CRREM carbon intensity | Annual actual vs. CRREM pathway | Below pathway threshold |

**Phase 3: Deviation Alerts**

| Alert Level | Trigger | Action | Notification |
|-------------|---------|--------|--------------|
| Information | EUI 5-10% above predicted for 3+ consecutive days | Log for trend review | Dashboard indicator |
| Warning | EUI 10-20% above predicted for 1+ week, or system COP degradation >10% | Review within 1 week | Email to facilities_manager |
| Critical | EUI >20% above predicted, or system failure detected, or baseload step change | Immediate investigation | SMS + email to facilities_manager + building_owner |
| Compliance | CRREM pathway exceedance projected within 2 years | Strategic review required | All stakeholders + quarterly report |

**Phase 4: Trend Reporting**

| Analysis | Frequency | Output |
|----------|-----------|--------|
| Daily energy profile | Daily | 24-hour load profile with baseload identification |
| Weekly performance summary | Weekly | Week-over-week EUI, system efficiency, weather comparison |
| Monthly benchmarking update | Monthly | EUI vs. benchmark, degree-day-adjusted performance |
| Quarterly DEC update | Quarterly | Updated operational rating, trend direction |
| Annual performance review | Annual | Full-year EUI by end use, carbon intensity, CRREM update, savings achieved |

**Acceptance Criteria:**
- [ ] Data ingestion latency < 10 seconds from BMS read to database
- [ ] Performance tracking updates within 1 minute of data availability
- [ ] False positive alert rate < 5%
- [ ] CRREM pathway tracking updates with new annual data within 24 hours
- [ ] Trend reports auto-generated on schedule

### 5.5 Workflow 5: Certification Assessment Workflow

**Purpose:** Green building certification assessment for LEED, BREEAM, Energy Star, and DGNB.

**Phase 1: Scheme Selection**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Identify applicable schemes | Building type, location, project stage (new/existing), market requirements | Applicable certification schemes | <5 minutes |
| 1.2 | Select target certification | Stakeholder preference, cost-benefit | Selected scheme(s) and target level (e.g., LEED Gold, BREEAM Excellent) | <5 minutes |
| 1.3 | Map available credits | Building characteristics, existing performance | Achievable credits by category | <10 minutes (auto) |

**Phase 2: Credit Mapping**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Assess energy credits | Building energy performance, ASHRAE 90.1 baseline (LEED), EPBD performance (BREEAM) | Energy credit score estimate | <15 minutes (auto) |
| 2.2 | Assess IEQ credits | Indoor environment assessment, thermal comfort, air quality, daylighting | IEQ credit score estimate | <10 minutes (auto) |
| 2.3 | Assess materials credits | Whole life carbon, EPD data, recycled content, responsible sourcing | Materials credit score estimate | <10 minutes (auto) |
| 2.4 | Assess water credits | Water consumption, rainwater harvesting, grey water recycling | Water credit score estimate | <5 minutes (auto) |
| 2.5 | Map cross-cutting credits | Innovation, regional priority, management credits | Additional credit score | <5 minutes |

**Phase 3: Performance Calculation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Calculate LEED EA credits | ASHRAE 90.1 baseline comparison, renewable energy, commissioning | LEED EA credit total (out of 33 EA points) | <10 minutes (auto) |
| 3.2 | Calculate BREEAM Ene credits | EPBD performance, CO2 reduction, sub-metering, energy monitoring | BREEAM Ene credit total (out of 34 Ene credits) | <10 minutes (auto) |
| 3.3 | Calculate Energy Star score | Normalized source EUI, CBECS regression | Energy Star 1-100 score | <5 minutes (auto) |
| 3.4 | Calculate DGNB ENV criteria | Whole life carbon per EN 15978, lifecycle cost per EN 16627 | DGNB ENV score | <10 minutes (auto) |

**Phase 4: Certification Scorecard**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Aggregate scores | All credit assessments | Total score per scheme | <2 minutes (auto) |
| 4.2 | Determine certification level | Total score vs. thresholds | Predicted certification level (e.g., Gold, Excellent) | <1 minute (auto) |
| 4.3 | Identify gap-to-next-level | Current score vs. next threshold | Credits needed, improvement actions | <3 minutes (auto) |
| 4.4 | Generate scorecard | All results | Certification scorecard report (PDF/HTML/JSON) | <5 minutes (auto) |

**Acceptance Criteria:**
- [ ] LEED EA credit estimate within 2 points of official LEED assessment
- [ ] BREEAM Ene credit estimate within 1 credit of official BREEAM assessment
- [ ] Energy Star score within 3 points of Portfolio Manager calculation
- [ ] Scorecard clearly identifies achievable vs. stretch credits
- [ ] Gap-to-next-level includes specific actionable improvement measures

### 5.6 Workflow 6: Tenant Engagement Workflow

**Purpose:** Tenant-facing energy reporting with consumption breakdown and reduction guidance.

**Phase 1: Tenant Metering**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Configure tenant sub-metering | Sub-meter registry per tenant, common area meters | Tenant metering hierarchy | <15 minutes |
| 1.2 | Ingest tenant meter data | Sub-meter readings (automated or manual) | Tenant consumption time series | Ongoing (auto) |
| 1.3 | Allocate common area energy | Common area energy, allocation methodology (floor area, occupancy, metered) | Per-tenant common area allocation | <5 minutes (auto) |

**Phase 2: Consumption Reporting**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Calculate tenant EUI | Metered + allocated energy, tenant floor area | Tenant EUI (kWh/m2/yr) | <2 minutes (auto) |
| 2.2 | Compare against peers | Tenant EUI vs. other tenants in building, vs. sector benchmark | Peer comparison with percentile ranking | <2 minutes (auto) |
| 2.3 | Trend analysis | Month-over-month, year-over-year | Consumption trend, seasonal pattern, improvement trajectory | <2 minutes (auto) |
| 2.4 | Cost breakdown | Metered energy, tariff, common area charge | Monthly energy cost statement with breakdown | <2 minutes (auto) |

**Phase 3: Behaviour Recommendations**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Identify saving opportunities | Consumption pattern analysis, peer comparison | Tenant-specific reduction recommendations | <3 minutes (auto) |
| 3.2 | Generate tenant report | All results | Tenant energy report (PDF/HTML) | <3 minutes (auto) |
| 3.3 | Track engagement | Report distribution, feedback, action taken | Engagement dashboard | Ongoing |

**Acceptance Criteria:**
- [ ] Tenant metering supports hourly or half-hourly data
- [ ] Common area allocation methodology configurable (3 methods)
- [ ] Peer comparison anonymized (no tenant names in comparison)
- [ ] Tenant report generated monthly, auto-distributed via email

### 5.7 Workflow 7: Regulatory Compliance Workflow

**Purpose:** Multi-jurisdiction regulatory compliance management for building energy regulations.

**Phase 1: Obligation Mapping**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Identify applicable regulations | Building location, type, size, age, use, tenure | List of applicable regulations with requirements | <5 minutes (auto) |
| 1.2 | Map current compliance | Current EPC, building performance, existing certificates | Compliance status per regulation (compliant/gap/non-compliant) | <10 minutes (auto) |
| 1.3 | Identify upcoming obligations | Regulatory roadmap (EPBD timeline, MEES escalation, BPS deadlines) | Future obligation timeline with thresholds | <5 minutes (auto) |

**Phase 2: Compliance Assessment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | EPBD compliance check | Current EPC, building type, MEPS thresholds | EPBD compliance status (EPC validity, MEPS, renovation obligation) | <5 minutes (auto) |
| 2.2 | MEES compliance check | EPC rating, rental status, exemption eligibility | MEES compliance (pass/fail/exempt), minimum EPC threshold | <3 minutes (auto) |
| 2.3 | BPS compliance check | Building carbon intensity vs. BPS pathway | BPS compliance projection with timeline | <3 minutes (auto) |
| 2.4 | F-gas compliance check | Refrigerant inventory, GWP, charge size | F-gas compliance (leak checks, reporting, phase-down) | <3 minutes (auto) |
| 2.5 | DEC compliance check | Building type, floor area, public authority status | DEC requirement status, validity, display obligation | <2 minutes (auto) |

**Phase 3: Deadline Tracking**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Create compliance calendar | All obligations with deadlines | Unified compliance calendar with RAG status | <5 minutes (auto) |
| 3.2 | Set advance warnings | Deadlines, preparation lead times | 12, 6, 3, 1 month advance warnings | <2 minutes (auto) |
| 3.3 | Generate compliance report | All compliance results | Regulatory compliance report (PDF/HTML/JSON) | <5 minutes (auto) |

**Acceptance Criteria:**
- [ ] Covers EPBD, MEES (UK/EU), BPS (Fit for 55), F-gas, DEC requirements
- [ ] Jurisdiction-specific thresholds loaded for 10+ EU Member States + UK
- [ ] Deadline calendar auto-populated with advance warning alerts
- [ ] MEES exemption eligibility check automated (5-year payback test, listed building, etc.)

### 5.8 Workflow 8: nZEB Readiness Workflow

**Purpose:** Nearly Zero-Energy Building and Zero-Emission Building readiness assessment and pathway.

**Phase 1: Current Performance**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Assess current primary energy | Building assessment results | Current primary energy (kWh/m2/yr), CO2 (kgCO2/m2/yr) | <5 minutes (auto) |
| 1.2 | Assess current renewable fraction | Renewable systems assessment | Current renewable energy fraction (%) | <3 minutes (auto) |
| 1.3 | Assess envelope performance | Envelope engine results | Current fabric performance vs. nZEB targets | <3 minutes (auto) |
| 1.4 | Assess system efficiency | HVAC engine results | Current system efficiency vs. nZEB targets | <3 minutes (auto) |

**Phase 2: Gap Quantification**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Define nZEB/ZEB targets | National nZEB definition, EPBD ZEB requirements | Target primary energy, CO2, renewable fraction | <3 minutes |
| 2.2 | Calculate performance gaps | Current vs. target | Gaps in kWh/m2/yr, kgCO2/m2/yr, renewable % | <2 minutes (auto) |
| 2.3 | Identify gap-closure measures | Applicable retrofit measures to close each gap | Measure list per gap category (envelope, HVAC, renewables) | <5 minutes (auto) |
| 2.4 | Passive House / EnerPHit check | Space heating demand, air tightness, primary energy | Passive House criteria compliance status | <3 minutes (auto) |

**Phase 3: Retrofit Pathway**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Design least-cost nZEB pathway | Gap-closure measures, interaction modelling, financial parameters | Optimized retrofit package achieving nZEB | <10 minutes (auto) |
| 3.2 | Design ZEB pathway | nZEB package + additional renewables/electrification | Extended package achieving ZEB | <5 minutes (auto) |
| 3.3 | Financial analysis | Package costs, savings, incentives | NPV, IRR, payback for nZEB and ZEB pathways | <5 minutes (auto) |

**Phase 4: ZEB Roadmap**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Define implementation timeline | Measures, dependencies, budget, operational constraints | Staged implementation roadmap (3-5 stages) | <5 minutes |
| 4.2 | Project CRREM compliance | Retrofit pathway vs. CRREM trajectory | Year-by-year CRREM alignment with/without retrofit | <3 minutes (auto) |
| 4.3 | Assess Building Renovation Passport | Stage-by-stage passport per EPBD Article 12 | BRP document | <5 minutes (auto) |
| 4.4 | Generate nZEB readiness report | All results | nZEB/ZEB readiness report (PDF/HTML/JSON) | <5 minutes (auto) |

**Acceptance Criteria:**
- [ ] nZEB target uses correct national definition (varies by Member State)
- [ ] ZEB target aligned with EPBD recast Article 7 (zero-emission building)
- [ ] Passive House / EnerPHit criteria checked per PHI requirements
- [ ] CRREM pathway projection includes grid decarbonization trajectory
- [ ] Building Renovation Passport format per EPBD Article 12 template

---

## 6. Template Specifications

### 6.1 Template 1: EPC Report

**Purpose:** Energy Performance Certificate per EPBD with A-G rating, primary energy, CO2, and recommendations.

**Sections:**
- Building identification (address, type, floor area, reference number, assessor details)
- EPC rating (A-G graphic with primary energy value)
- Energy performance breakdown (heating, cooling, DHW, lighting, ventilation, renewables)
- CO2 emissions (kgCO2/m2/yr)
- Reference building comparison (where applicable per national methodology)
- Building envelope summary (U-values, air tightness)
- Services summary (heating, cooling, DHW, lighting, ventilation systems)
- Renewable energy systems
- Improvement recommendations (ranked by cost-effectiveness with estimated savings and payback)
- Assessor certification and methodology declaration
- Validity period (10 years from date of issue)

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 4-8 pages (residential), 8-15 pages (commercial)

### 6.2 Template 2: DEC Report

**Purpose:** Display Energy Certificate for operational rating based on metered energy consumption.

**Sections:**
- Building identification
- Operational rating (A-G based on actual vs. benchmark energy consumption)
- Annual energy consumption (kWh) by carrier
- Floor area (m2 treated floor area)
- Benchmark comparison (CIBSE TM46 typical and good practice)
- Weather-corrected EUI
- Year-on-year trend (previous 3 years where available)
- Advisory report reference (with recommendation summary)
- Assessor details and valid dates

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 2-4 pages

### 6.3 Template 3: Building Assessment Report

**Purpose:** Comprehensive building energy assessment covering envelope, systems, benchmarks, and recommendations.

**Sections:**
- Executive summary (key findings, EPC rating, EUI, top 5 recommendations)
- Building description (type, age, construction, floor area, occupancy)
- Building envelope assessment (U-values, thermal bridges, air tightness, condensation risk)
- HVAC systems assessment (heating efficiency, cooling efficiency, ventilation, controls)
- DHW system assessment (demand, efficiency, solar contribution)
- Lighting assessment (LENI, LPD, daylight factor, controls)
- Renewable energy systems (PV yield, solar thermal fraction, renewable fraction)
- Energy balance (monthly energy needs, delivered energy, primary energy)
- Benchmarking (EUI, Energy Star, CRREM pathway, DEC rating)
- Indoor environment assessment (thermal comfort, air quality, overheating risk)
- Improvement recommendations (60+ measure assessment with prioritization)
- Financial analysis (per-measure and package NPV, IRR, payback)
- Staged retrofit roadmap
- Appendices (calculation details, data sources, assumptions)

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 30-80 pages

### 6.4 Template 4: Retrofit Recommendation Report

**Purpose:** Retrofit business case with measure details, savings, costs, and staged roadmap.

**Sections:**
- Executive summary (total savings potential, investment required, average payback)
- Current performance baseline (EUI, primary energy, CO2, EPC rating)
- Performance targets (nZEB, ZEB, CRREM, certification)
- Measure-by-measure analysis (description, savings, cost, payback, NPV, IRR per measure)
- Interaction matrix (measure-measure interactions quantified)
- Recommended packages (3-5 investment levels)
- Package financial analysis (NPV, IRR, lifecycle cost, sensitivity)
- Staged implementation roadmap with timeline
- CRREM pathway trajectory pre/post retrofit
- Financing options (grants, green loans, ESCO, EPC)
- Risk register (delivery risk, performance risk, market risk)
- Next steps and implementation plan

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 20-50 pages

### 6.5 Template 5: Building Benchmark Report

**Purpose:** Peer comparison dashboard with EUI benchmarks, Energy Star, and CRREM pathway.

**Sections:**
- Building identification and type classification
- EUI comparison (site EUI, source EUI, weather-normalized)
- Benchmark comparison (CIBSE TM46 typical/good/best practice)
- Energy Star score estimation (1-100) with certification eligibility
- CRREM pathway chart (current position, pathway, stranding year, scenarios)
- DEC operational rating
- Peer comparison (percentile ranking within building type and climate zone)
- End-use breakdown comparison (heating, cooling, DHW, lighting vs. benchmarks)
- Year-on-year performance trend
- Improvement potential (gap-to-best-practice in kWh/m2 and EUR/m2)

**Output Formats:** MD, HTML, JSON
**Typical Length:** 8-15 pages

### 6.6 Template 6: Certification Scorecard

**Purpose:** LEED/BREEAM/Energy Star scorecard with credit-by-credit assessment.

**Sections:**
- Target certification summary (scheme, level, total points)
- Credit-by-credit assessment table (credit ID, name, available points, estimated points, confidence, evidence)
- Energy category detail (LEED EA / BREEAM Ene)
- IEQ category detail (LEED EQ / BREEAM Hea)
- Materials category detail (LEED MR / BREEAM Mat)
- Water category detail (LEED WE / BREEAM Wat)
- Innovation/regional detail
- Gap analysis (credits needed for target level, actions per credit)
- Certification roadmap (timeline, cost, evidence requirements)
- Cross-scheme comparison (if multiple certifications assessed)

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 15-30 pages

### 6.7 Template 7: Tenant Energy Report

**Purpose:** Tenant-facing energy report with consumption breakdown and reduction tips.

**Sections:**
- Tenant identification and demise
- Monthly energy consumption (electricity, gas, heat)
- Energy cost breakdown (metered energy + common area allocation)
- Comparison vs. previous month and previous year
- Peer comparison (anonymized ranking within building)
- Carbon footprint (Scope 1+2 from building energy)
- End-use breakdown (where sub-metering allows: lighting, power, HVAC)
- Reduction recommendations (behaviour-based, no-cost/low-cost)
- Engagement score and trend

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 2-4 pages

### 6.8 Template 8: Building Dashboard

**Purpose:** Real-time building performance dashboard for continuous monitoring.

**Dashboard Panels:**
- Current EUI (real-time, rolling 24-hour, rolling 7-day, rolling 30-day)
- EUI vs. benchmark overlay (actual vs. CIBSE TM46 typical/good/best)
- Energy by end use (heating, cooling, DHW, lighting, small power, other)
- System efficiency tracking (heat pump COP/SPF, chiller SEER, AHU SFP)
- Degree-day analysis (actual HDD/CDD vs. long-term average, weather-adjusted EUI)
- Baseload tracking (overnight and weekend minimum demand)
- Peak demand tracking (maximum demand and time of occurrence)
- Alert status (active warnings, critical alerts, compliance alerts)
- Carbon intensity (kgCO2/m2/yr actual vs. CRREM pathway)
- Tenant comparison (building-wide and per-tenant, anonymized)

**Output Formats:** MD, HTML, JSON

### 6.9 Template 9: Regulatory Compliance Report

**Purpose:** EPBD/MEES/BPS compliance summary with obligation status and deadlines.

**Sections:**
- Building identification and classification
- Regulatory jurisdiction and applicable regulations
- EPBD compliance status (EPC validity, MEPS compliance, renovation obligations)
- MEES compliance status (minimum EPC rating, exemption status)
- BPS compliance status (carbon intensity vs. BPS pathway)
- F-gas compliance status (refrigerant inventory, leak check schedule, GWP phase-down)
- DEC compliance status (requirement, validity, display obligation)
- National building regulations compliance (Part L / equivalent)
- Upcoming deadlines with RAG status (12-month forward view)
- Compliance action items with responsible person and due date
- Penalty/risk exposure for non-compliance

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 8-15 pages

### 6.10 Template 10: Whole Life Carbon Report

**Purpose:** Embodied + operational carbon report per EN 15978 with lifecycle stage breakdown.

**Sections:**
- Executive summary (total WLC, kgCO2e/m2, embodied vs. operational split)
- Building description and study parameters (reference study period, system boundary)
- Product stage carbon (A1-A3) by building element (substructure, superstructure, envelope, services)
- Construction stage carbon (A4-A5)
- Use stage carbon (B1-B7 including B6 operational energy and B7 operational water)
- End-of-life carbon (C1-C4)
- Beyond system boundary (Module D credits)
- Whole life carbon breakdown chart (stacked bar by lifecycle stage)
- Budget comparison (RIBA 2030, LETI, GLA, DGNB targets)
- Material hotspot analysis (top 10 materials by embodied carbon)
- Material substitution opportunities (alternatives with carbon saving and cost impact)
- Biogenic carbon accounting (timber and bio-based materials)
- Sensitivity analysis (building lifetime, grid decarbonization, material choices)

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 15-30 pages

---

## 7. Integration Specifications

### 7.1 Integration 1: Building Assessment Orchestrator

**Purpose:** Master orchestration pipeline for all PACK-032 engines.

**DAG Pipeline (12 phases):**

```
Phase 1: Building Setup (setup_wizard)
  |
Phase 2: Data Ingestion (data_building_bridge, bms_integration_bridge, weather_data_bridge)
  |
Phase 3: Envelope Analysis (building_envelope_engine)
  |
Phase 4: HVAC Assessment (hvac_assessment_engine)
  |
Phase 5: Subsystem Assessments [conditional, parallel]
  |-- 5a: DHW (domestic_hot_water_engine) [if DHW system present]
  |-- 5b: Lighting (lighting_assessment_engine) [always]
  |-- 5c: Renewables (renewable_integration_engine) [if renewables present or assessed]
  |
Phase 6: EPC Rating (epc_rating_engine)
  |
Phase 7: Benchmarking (building_benchmark_engine)
  |
Phase 8: Retrofit Analysis (retrofit_analysis_engine)
  |
Phase 9: Indoor Environment [conditional]
  |-- 9a: IEQ Assessment (indoor_environment_engine) [if IEQ in scope]
  |
Phase 10: Whole Life Carbon [conditional]
  |-- 10a: WLC Assessment (whole_life_carbon_engine) [if WLC in scope]
  |
Phase 11: Certification Assessment [conditional]
  |-- 11a: LEED/BREEAM/Energy Star (certification_assessment_workflow) [if certification in scope]
  |
Phase 12: Report Generation (all applicable templates)
```

**Orchestrator Features:**
- Conditional phase execution based on building type and scope
- Parallel execution of independent subsystem assessments (Phase 5)
- Phase retry with exponential backoff (3 retries, 30s/60s/120s)
- SHA-256 provenance hash per phase output
- Progress tracking with percentage completion per phase
- Partial result caching (resume from last completed phase)
- Configurable timeout per phase (default 10 minutes, max 30 minutes)

### 7.2 Integration 2: MRV Building Bridge

**Purpose:** Route to building-specific MRV agents for GHG emissions calculation.

**MRV Agent Mapping:**

| MRV Agent | Purpose | PACK-032 Connection |
|-----------|---------|---------------------|
| GL-MRV-BLD-001 | Gas/oil combustion emissions (Scope 1) | Boiler fuel consumption -> Scope 1 CO2 |
| GL-MRV-BLD-002 | Purchased electricity emissions (Scope 2) | Grid electricity -> Scope 2 CO2 (location + market) |
| GL-MRV-BLD-003 | Purchased heat/steam emissions (Scope 2) | District heat -> Scope 2 CO2 |
| GL-MRV-BLD-004 | Purchased cooling emissions (Scope 2) | District cooling -> Scope 2 CO2 |
| GL-MRV-BLD-005 | Refrigerant leakage emissions (Scope 1) | F-gas inventory -> Scope 1 CO2e |
| GL-MRV-BLD-006 | On-site renewable generation | Solar PV, solar thermal -> Scope 2 avoidance |
| GL-MRV-BLD-007 | Building operational carbon | Annual operational CO2/m2 for CRREM |
| GL-MRV-BLD-008 | Embodied carbon (construction/renovation) | Material quantities -> Scope 3 Cat 1/2 |

### 7.3 Integration 3: Data Building Bridge

**Purpose:** Route to DATA agents for building data ingestion and quality.

**DATA Agent Mapping:**

| DATA Agent | Purpose | PACK-032 Connection |
|-----------|---------|---------------------|
| DATA-001 | PDF extraction | EPC certificates, survey reports, equipment schedules |
| DATA-002 | Excel/CSV normalization | Utility bills, BMS exports, meter readings, asset registers |
| DATA-003 | ERP/Finance connector | Energy procurement, maintenance costs, capital expenditure |
| DATA-010 | Data quality profiler | Building data completeness, consistency, accuracy assessment |
| DATA-014 | Time series gap filler | BMS data gaps, meter data gaps |
| DATA-016 | Data freshness monitor | BMS feed currency, weather data currency |

### 7.4 Integration 4: EPBD Compliance Bridge

**Purpose:** EPBD Directive (EU) 2024/1275 compliance integration.

**Key Functions:**
- EPC validity tracking (10-year validity, renewal reminders at 12/6/3 months)
- Minimum Energy Performance Standards monitoring (worst-performing building thresholds per EPBD Article 9)
- Renovation obligation assessment (major renovation triggers per Article 8)
- Zero-emission building timeline (new buildings from 2030, public buildings from 2028)
- Building Renovation Passport support (per Article 12)
- Solar energy obligation (per Article 10, new buildings and major renovations)
- EPC register submission formatting (per national register requirements)

### 7.5 Integration 5: BMS Integration Bridge

**Purpose:** Building Management System data integration.

**Supported Protocols:**
- BACnet/IP (ASHRAE 135)
- Modbus TCP/RTU
- KNX/EIB
- LonWorks
- OPC-UA (IEC 62541)
- MQTT (for IoT sensors)

**Data Points Ingested:**
- Zone temperatures (air, mean radiant, setpoint)
- HVAC system status (on/off, mode, capacity)
- Energy meters (electricity, gas, heat, cooling)
- Equipment run hours and status
- Ventilation rates and CO2 levels
- Humidity levels
- Solar irradiance on building (if weather station present)
- Lighting levels and occupancy sensors

### 7.6 Integration 6: Weather Data Bridge

**Purpose:** Weather data integration for energy calculation and normalization.

**Data Sources:**
- Meteonorm / PVGIS for TMY (Typical Meteorological Year) data
- Meteostat / Open-Meteo for historical and real-time European weather
- NOAA ISD for global coverage
- National weather services (Met Office UK, DWD Germany, Meteo-France)

**Output:**
- Hourly weather data (dry-bulb, wet-bulb, relative humidity, wind speed, solar radiation)
- Heating Degree Days (HDD) with configurable base temperature (default 15.5C)
- Cooling Degree Days (CDD) with configurable base temperature (default 22C)
- Solar radiation by orientation and tilt (for PV and solar thermal)
- Climate zone classification (Koppen, ASHRAE, national)

### 7.7 Integration 7: Certification Bridge

**Purpose:** Green building certification system integration.

**Supported Certifications:**
- LEED Online (USGBC API for credit tracking, documentation submission)
- BREEAM Projects (BRE Global data exchange)
- Energy Star Portfolio Manager (EPA API for building benchmarking and certification)
- DGNB System (DGNB data exchange format)
- NABERS (rating calculator integration)

**Data Exchange:**
- Building characteristics and performance data
- Credit documentation and evidence
- Score calculation verification
- Certification status and expiry tracking

### 7.8 Integration 8: Grid Carbon Bridge

**Purpose:** Grid carbon intensity data for operational carbon calculation and demand response.

**Data Sources:**
- DEFRA (UK government emission factors, annual)
- UBA (German Umweltbundesamt emission factors)
- ADEME (French emission factors)
- ISPRA (Italian emission factors)
- AIB (European Residual Mix factors for Scope 2 market-based)
- ElectricityMap API (real-time grid carbon intensity, 30+ countries)
- National grid operator forecasts (day-ahead carbon intensity)

**Output:**
- Annual average grid emission factor (kgCO2/kWh) per country
- Monthly/seasonal grid emission factor variation
- Half-hourly marginal emission factor (for demand response optimization)
- Grid decarbonization trajectory (10-30 year projection per national climate plan)
- Residual mix factor for Scope 2 market-based method

### 7.9 Integration 9: Property Registry Bridge

**Purpose:** Building and property database integration.

**Data Sources:**
- National cadastral databases (footprint, height, floors, year of construction)
- EPC register lookup (existing EPCs, historical ratings, lodgement data)
- Land Registry (ownership, tenure, leasehold/freehold)
- Building control records (approved plans, Part L compliance, completion certificates)
- Listed building registers (heritage constraints, conservation area status)
- TABULA/EPISCOPE building typology (age-based default construction data)

### 7.10 Integration 10: Health Check

**Purpose:** System verification for all PACK-032 components.

**20 Verification Categories:**

| # | Category | Checks |
|---|----------|--------|
| 1 | Engine availability | All 10 engines respond to health ping |
| 2 | Workflow availability | All 8 workflows respond to health ping |
| 3 | Template availability | All 10 templates generate test output |
| 4 | Database connectivity | PostgreSQL connection, migration status (V191-V200) |
| 5 | Redis cache | Cache connectivity and response time |
| 6 | MRV bridge | Connection to MRV building agents (BLD-001 through BLD-008) |
| 7 | Data bridge | Connection to DATA agents (001, 002, 003, 010, 014, 016) |
| 8 | Foundation bridge | Connection to FOUND agents (001-010) |
| 9 | BMS connectivity | Protocol adapters responding (BACnet, Modbus, KNX) |
| 10 | Weather data feed | Latest weather data within 24 hours |
| 11 | Grid carbon data | Emission factor data current and loaded |
| 12 | EPC register | API connectivity and data currency |
| 13 | Certification APIs | LEED/BREEAM/Energy Star API connectivity |
| 14 | CRREM data | Pathway data loaded for all building types and countries |
| 15 | Material database | EPD/ICE embodied carbon data loaded |
| 16 | Authentication | JWT RS256 token issuance/validation |
| 17 | Authorization | RBAC permission checks for all 5 roles |
| 18 | Encryption | AES-256-GCM encrypt/decrypt test |
| 19 | Audit trail | SEC-005 audit logging operational |
| 20 | Provenance | SHA-256 hash generation/verification |

### 7.11 Integration 11: Setup Wizard

**Purpose:** Guided 8-step building configuration for new deployments.

**Steps:**

| Step | Configuration | Inputs Required |
|------|--------------|-----------------|
| 1. Building Profile | Name, address, type, year of construction, floor area (GIA, NIA, TFA), number of floors, building height | Basic building information |
| 2. Envelope Construction | Wall types, roof type, ground floor type, windows (type, area, orientation), construction age/details | Construction details or age-based defaults |
| 3. HVAC Systems | Heating (type, fuel, capacity, age), cooling (type, refrigerant, capacity), ventilation (type, SFP, heat recovery) | System inventory |
| 4. Lighting | Luminaire types by zone, controls, operating hours, daylight provision | Lighting schedule |
| 5. DHW System | System type, fuel, storage, distribution, solar thermal (if present) | DHW system details |
| 6. Renewables | Solar PV (kWp, orientation, tilt), solar thermal (area, type), other renewables | Renewable system details |
| 7. Metering Infrastructure | Main meters, sub-meters, BMS data sources, data frequency | Meter inventory |
| 8. Regulatory Jurisdiction | Country, region, applicable regulations, EPC methodology, certification targets | Regulatory configuration |

### 7.12 Integration 12: CRREM Pathway Bridge

**Purpose:** CRREM (Carbon Risk Real Estate Monitor) decarbonization pathway compliance integration.

**Key Functions:**
- Pathway lookup by building type, country, and scenario (1.5C, 2.0C)
- Annual carbon intensity threshold (kgCO2/m2/yr) from CRREM database
- Current building carbon intensity calculation (from operational data)
- Stranding year calculation (BAU scenario: when building exceeds pathway)
- Retrofit scenario modelling (post-retrofit stranding year delay/avoidance)
- Grid decarbonization projection (reducing grid carbon intensity over time)
- Portfolio-level CRREM analysis (aggregate stranding risk across portfolio)
- Annual CRREM data update integration (new pathways, country updates)

**CRREM Building Types (Selected):**

| CRREM Type | Description | 1.5C 2025 Target (kgCO2/m2) | 1.5C 2030 Target (kgCO2/m2) |
|-----------|-------------|-------------------------------|-------------------------------|
| Office | General office | 30-55 (varies by country) | 20-40 |
| Retail (high street) | Shopping / retail | 45-70 | 30-50 |
| Hotel | Hotel / hospitality | 40-65 | 25-45 |
| Residential | Multi-family residential | 20-40 | 12-28 |
| Healthcare | Hospital / clinic | 60-90 | 40-65 |
| Education | School / university | 25-45 | 15-30 |
| Industrial | Light industrial / warehouse | 20-35 | 12-22 |
| Logistics | Distribution / logistics | 15-25 | 8-15 |

Note: Values are illustrative ranges; actual CRREM thresholds vary by country and are updated annually.

---

## 8. Preset Specifications

### 8.1 Preset 1: Commercial Office

**Building Type:** Commercial office (open-plan, cellular, mixed, high-rise)
**Energy Profile:** Cooling-dominated in summer, internal gains dominant

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | Low priority (small DHW demand in offices) |
| Lighting assessment | High priority (20-30% of energy) |
| Cooling assessment | High priority (internal gains driven) |
| Primary energy carrier | Electricity (60-80%), Natural gas (15-30%), District heat (0-20%) |
| Key end uses | Cooling (25-35%), lighting (20-30%), small power (15-25%), heating (15-25%) |
| EUI default benchmark | CIBSE TM46 Type 1/2 (120-300 kWh/m2/yr depending on AC) |
| CRREM type | Office |
| Typical EPC rating (existing stock) | D-F |
| nZEB target primary energy | 50-90 kWh/m2/yr (varies by MS) |
| Certification focus | LEED, BREEAM, Energy Star |
| Typical retrofit savings potential | 25-45% of total energy |

### 8.2 Preset 2: Retail Building

**Building Type:** Retail / Shopping Centre / Food Retail
**Energy Profile:** Lighting and refrigeration intensive, long operating hours

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | Low priority (unless food service areas) |
| Lighting assessment | High priority (25-35% of energy, 12-16 hrs/day operation) |
| Cooling/refrigeration | High priority (food retail: 40-60% for refrigeration) |
| Primary energy carrier | Electricity (70-90%), Natural gas (10-25%) |
| Key end uses | Lighting (25-35%), refrigeration (0-50%, food retail), HVAC (20-30%), small power (10-15%) |
| EUI default benchmark | CIBSE TM46 Type 5/6 (200-700 kWh/m2/yr) |
| CRREM type | Retail (high street) / Shopping centre |
| Typical EPC rating (existing stock) | D-G |
| Certification focus | BREEAM, LEED |
| Typical retrofit savings potential | 20-40% |

### 8.3 Preset 3: Hotel / Hospitality

**Building Type:** Hotel (business, resort, serviced apartments)
**Energy Profile:** 24/7 operation, high DHW demand, variable occupancy

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | High priority (100-200 L/room/day) |
| Lighting assessment | Medium priority (guest room + common area) |
| HVAC assessment | High priority (individual room controls, 24/7 common areas) |
| Primary energy carrier | Electricity (40-55%), Natural gas (30-45%), LPG/oil (5-15%) |
| Key end uses | Heating (20-30%), DHW (20-30%), cooling (10-20%), lighting (15-20%), kitchen (10-20%) |
| EUI default benchmark | CIBSE TM46 Type 7 (250-400 kWh/m2/yr) |
| CRREM type | Hotel |
| Typical EPC rating (existing stock) | D-F |
| Certification focus | BREEAM, Green Key, LEED |
| Typical retrofit savings potential | 20-35% |

### 8.4 Preset 4: Healthcare Facility

**Building Type:** Hospital, clinic, care home
**Energy Profile:** 24/7 operation, high ventilation, sterilization, medical equipment

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | High priority (high demand for washing, sterilization) |
| Lighting assessment | Medium priority (24/7 lit areas, operating theatres 1000 lux) |
| HVAC assessment | Critical (6-15 ACH for theatres, infection control, clean rooms) |
| Indoor environment | Critical (strict IEQ for patient safety) |
| Primary energy carrier | Electricity (45-55%), Natural gas (35-45%), Steam (5-15%) |
| Key end uses | HVAC (35-50%), lighting (15-20%), DHW (10-15%), medical equipment (10-15%), sterilization (5-10%) |
| EUI default benchmark | CIBSE TM46 Type 3/4 (350-500 kWh/m2/yr) |
| CRREM type | Healthcare |
| Typical EPC rating (existing stock) | D-G |
| Certification focus | BREEAM Healthcare |
| Typical retrofit savings potential | 15-30% |

### 8.5 Preset 5: Education Building

**Building Type:** School, university, nursery
**Energy Profile:** Intermittent occupancy, heating-dominated, high ventilation

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | Low priority (minimal DHW in schools) |
| Lighting assessment | High priority (classrooms, lecture halls, libraries) |
| HVAC assessment | High priority (high ventilation per pupil, intermittent operation) |
| Indoor environment | High priority (BB101 ventilation requirements, thermal comfort for learning) |
| Primary energy carrier | Electricity (40-55%), Natural gas (35-50%), Biomass (0-10%) |
| Key end uses | Heating (40-55%), lighting (15-25%), ventilation (10-15%), small power (10-15%), sports/pool (0-20%) |
| EUI default benchmark | CIBSE TM46 Type 9/10/11 (130-250 kWh/m2/yr) |
| CRREM type | Education |
| Typical EPC rating (existing stock) | D-F |
| Certification focus | BREEAM Education, DEC (mandatory for state schools) |
| Typical retrofit savings potential | 25-40% |

### 8.6 Preset 6: Residential Multifamily

**Building Type:** Apartment block, social housing, co-living
**Energy Profile:** Heating-dominated, DHW significant, individual unit metering

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | High priority (15-25% of energy per dwelling) |
| Lighting assessment | Medium priority (common areas + default per dwelling) |
| HVAC assessment | High priority (communal heating, individual heat pumps, or district heat) |
| Primary energy carrier | Electricity (30-50%), Natural gas (30-50%), District heat (0-30%) |
| Key end uses | Heating (50-70%), DHW (15-25%), lighting/appliances (15-25%) |
| EUI default benchmark | National dwelling stock average (100-200 kWh/m2/yr) |
| EPC methodology | SAP/RdSAP (UK), DPE (FR), GEG (DE), APE (IT) per dwelling |
| CRREM type | Residential |
| Typical EPC rating (existing stock) | D-G (pre-1990 stock often E-G) |
| Certification focus | Passive House, EnerPHit, national green building (Code for Sustainable Homes equivalent) |
| Typical retrofit savings potential | 30-60% (especially pre-1980 stock) |

### 8.7 Preset 7: Mixed-Use Development

**Building Type:** Residential + commercial, live-work, mixed retail/office/residential
**Energy Profile:** Multiple use zones, shared plant, complex metering

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | High priority (residential zones) |
| Lighting assessment | High priority (commercial zones) |
| HVAC assessment | High priority (shared plant, multiple zones) |
| Primary energy carrier | Variable by zone (commercial: electricity-heavy; residential: heat-heavy) |
| Key end uses | Varies by zone mix |
| EPC methodology | Separate EPC per zone/unit, shared plant energy allocation |
| CRREM type | Composite (area-weighted across types) |
| Certification focus | BREEAM (commercial), national standards (residential), Energy Star (where applicable) |
| Special considerations | Sub-metering per tenant/unit, common area allocation, CHP/district heat allocation between uses |

### 8.8 Preset 8: Public Sector Building

**Building Type:** Government, library, museum, community centre
**Energy Profile:** Variable public access, heritage constraints, DEC mandatory

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| DHW assessment | Low-medium priority |
| Lighting assessment | High priority (display lighting in museums, reading areas in libraries) |
| HVAC assessment | High priority (public access = variable occupancy, museums = strict environmental control) |
| Indoor environment | Critical for museums (temperature 20 +/- 2C, RH 50 +/- 5% for collections) |
| Primary energy carrier | Electricity (45-60%), Natural gas (30-45%), District heat (0-15%) |
| Key end uses | HVAC (30-50%), lighting (20-30%), IT/servers (10-15%), DHW (5-10%) |
| EUI default benchmark | CIBSE TM46 Type 6/8 (varies significantly by sub-type) |
| DEC requirement | Mandatory for public buildings >250 m2 (EPBD Article 13) |
| Heritage constraints | Listed building restrictions on external insulation, window replacement, solar PV |
| Certification focus | DEC (mandatory), BREEAM In-Use |
| Typical retrofit savings potential | 15-30% (constrained by heritage status) |

---

## 9. Agent Dependencies

### 9.1 MRV Building Agents (8)

Dedicated building MRV agents via `mrv_building_bridge.py`:
- **GL-MRV-BLD-001**: Gas/oil combustion emissions (Scope 1) from building heating and DHW
- **GL-MRV-BLD-002**: Purchased electricity emissions (Scope 2, location-based and market-based)
- **GL-MRV-BLD-003**: Purchased heat/steam emissions (Scope 2, district heating)
- **GL-MRV-BLD-004**: Purchased cooling emissions (Scope 2, district cooling)
- **GL-MRV-BLD-005**: Refrigerant leakage emissions (Scope 1, F-gas)
- **GL-MRV-BLD-006**: On-site renewable generation (Scope 2 offset)
- **GL-MRV-BLD-007**: Building operational carbon (annual kgCO2/m2 for CRREM tracking)
- **GL-MRV-BLD-008**: Embodied carbon for construction/renovation (Scope 3 Category 1/2)

### 9.2 Decarbonization Agents (2)

- **GL-DECARB-BLD-001**: Building-level decarbonization pathway modelling (electrification, heat pump scenarios)
- **GL-DECARB-BLD-002**: Portfolio-level building decarbonization strategy (prioritization across building stock)

### 9.3 Data Agents (20)

All 20 AGENT-DATA agents via `data_building_bridge.py`, with primary relevance for:
- **DATA-001 PDF Extractor**: EPC certificates, survey reports, equipment schedules
- **DATA-002 Excel/CSV Normalizer**: Utility bills, BMS exports, meter readings
- **DATA-003 ERP/Finance Connector**: Energy procurement, maintenance costs
- **DATA-010 Data Quality Profiler**: Building data completeness and accuracy assessment
- **DATA-014 Time Series Gap Filler**: BMS data gaps, meter data gaps
- **DATA-016 Data Freshness Monitor**: BMS and weather data currency monitoring

### 9.4 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema validation, unit normalization (critical for building energy units: kWh, kWh/m2, W/m2.K, m3/hr, L/s, Pa, ACH), assumptions registry, citations, access control, etc.

### 9.5 Application Dependencies

- **GL-GHG-APP**: GHG inventory management for building-related emissions
- **GL-CSRD-APP**: ESRS E1-5 energy consumption and mix disclosure for building portfolios
- **GL-CDP-APP**: CDP C7 Energy and C8 Energy-related emissions for buildings
- **GL-Taxonomy-APP**: EU Taxonomy climate mitigation criteria for building renovation activities

### 9.6 Optional Pack Dependencies

- **PACK-031 Industrial Energy Audit Pack**: Cross-reference for mixed industrial/building sites; shared weather normalization, degree-day calculation, and M&V methodologies
- **PACK-021 Net Zero Starter Pack**: Baseline emissions linked to building energy, reduction pathways
- **PACK-022 Net Zero Acceleration Pack**: Advanced decarbonization scenarios including building retrofit
- **PACK-023 SBTi Alignment Pack**: SBTi target-linked building energy efficiency programs

---

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| U-value calculation per building element | <1 second |
| Full envelope assessment (50 elements, 100 windows) | <30 seconds |
| EPC calculation (monthly method) per building | <30 seconds |
| EPC calculation (hourly method) per building | <5 minutes |
| Batch EPC processing (100 dwellings) | <15 minutes |
| HVAC assessment per building | <15 seconds |
| DHW assessment per building | <10 seconds |
| LENI calculation per building (50 zones) | <15 seconds |
| PV yield estimation per building | <10 seconds |
| Benchmarking (single building, all frameworks) | <15 seconds |
| CRREM pathway assessment per building | <5 seconds |
| Retrofit analysis (60+ measures, interaction modelling) | <3 minutes |
| Indoor environment assessment (PMV/PPD, IAQ, overheating) | <30 seconds |
| Whole life carbon calculation (100 materials, 60-year study) | <2 minutes |
| Full initial building assessment workflow | <45 minutes |
| EPC generation workflow | <30 minutes |
| Real-time data processing latency | <10 seconds |
| Memory ceiling | 4096 MB |
| Cache hit target | 70% |
| Max buildings per deployment | 10,000 |
| Max zones per building | 500 |
| Max concurrent assessments | 20 |

---

## 11. Security Requirements

- JWT RS256 authentication
- RBAC with 5 roles: `building_assessor`, `facilities_manager`, `portfolio_manager`, `tenant`, `admin`
- Building-level access control (users see only buildings assigned to their role)
- AES-256-GCM encryption at rest for all building data, energy data, and assessment results
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs (U-values, EPC ratings, benchmarks, retrofit analyses)
- Full audit trail per SEC-005 (who changed what, when, with provenance chain)
- BMS credentials encrypted via Vault (SEC-006)
- Certification API keys encrypted via Vault
- Read-only mode for tenants (consumption data view only, no configuration access)
- Data retention: minimum 10 years for EPC records (per EPBD), 7 years for energy data

**RBAC Permission Matrix:**

| Permission | building_assessor | facilities_manager | portfolio_manager | tenant | admin |
|------------|------------------|-------------------|-------------------|--------|-------|
| Create/edit building | Yes | No | No | No | Yes |
| Upload building data | Yes | Yes | No | No | Yes |
| Run building assessment | Yes | Yes | No | No | Yes |
| Generate EPC | Yes | No | No | No | Yes |
| View assessment results | Yes | Yes | Yes | No | Yes |
| View tenant energy data | No | Yes | Yes | Yes (own) | Yes |
| Run retrofit analysis | Yes | Yes | Yes | No | Yes |
| Approve retrofit plan | No | Yes | Yes | No | Yes |
| View CRREM pathway | Yes | Yes | Yes | No | Yes |
| Export data | Yes | Yes | Yes | Yes (own) | Yes |
| Manage users | No | No | No | No | Yes |
| View all buildings | No | No | Yes | No | Yes |
| Delete records | No | No | No | No | Yes |

---

## 12. Database Migrations

Inherits platform migrations V001-V190. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V191__pack032_building_assessment_001 | `bea_buildings`, `bea_building_zones`, `bea_building_profiles`, `bea_regulatory_jurisdictions` | Building registry with type/age/area, thermal zones with use type and schedule, building profiles for operational parameters, and regulatory jurisdiction configuration |
| V192__pack032_building_assessment_002 | `bea_envelope_elements`, `bea_construction_layers`, `bea_thermal_bridges`, `bea_windows`, `bea_air_tightness` | Building envelope elements with U-values, construction layers with material properties, thermal bridge psi-values, window performance data (Uw, g-value), and air tightness test results |
| V193__pack032_building_assessment_003 | `bea_hvac_systems`, `bea_heating_systems`, `bea_cooling_systems`, `bea_ventilation_systems`, `bea_refrigerants` | HVAC system inventory with type/capacity/efficiency, heating system seasonal efficiency, cooling SEER/EER, ventilation SFP and heat recovery, and refrigerant tracking for F-gas compliance |
| V194__pack032_building_assessment_004 | `bea_dhw_systems`, `bea_solar_thermal`, `bea_lighting_zones`, `bea_luminaire_schedule`, `bea_lighting_controls` | DHW systems with demand and efficiency, solar thermal collector data and f-chart results, lighting zones with LENI calculation, luminaire inventory, and lighting controls assessment |
| V195__pack032_building_assessment_005 | `bea_renewable_systems`, `bea_pv_arrays`, `bea_solar_thermal_arrays`, `bea_renewable_assessments` | Renewable energy system records, PV array details (kWp, orientation, tilt, yield), solar thermal array performance, and renewable fraction calculations |
| V196__pack032_building_assessment_006 | `bea_epc_records`, `bea_epc_calculations`, `bea_dec_records`, `bea_energy_balances` | EPC records with rating/primary energy/CO2/validity, detailed EPC calculation results per national methodology, DEC operational ratings, and monthly/annual energy balance data |
| V197__pack032_building_assessment_007 | `bea_benchmarks`, `bea_crrem_assessments`, `bea_energy_star_scores`, `bea_peer_comparisons` | Benchmarking records with EUI and weather-normalized EUI, CRREM pathway assessments with stranding year, Energy Star score estimates, and peer comparison rankings |
| V198__pack032_building_assessment_008 | `bea_retrofit_assessments`, `bea_retrofit_measures`, `bea_retrofit_packages`, `bea_retrofit_roadmaps`, `bea_certification_assessments` | Retrofit assessment records, individual measure analysis (savings, cost, payback), optimized retrofit packages, staged roadmaps, and certification scorecard records (LEED/BREEAM/Energy Star) |
| V199__pack032_building_assessment_009 | `bea_ieq_assessments`, `bea_thermal_comfort`, `bea_air_quality`, `bea_overheating`, `bea_wlc_assessments`, `bea_embodied_carbon`, `bea_material_substitutions` | Indoor environment quality assessments (PMV/PPD, IAQ, overheating), and whole life carbon records per EN 15978 (embodied, operational, end-of-life, Module D), with material substitution analysis |
| V200__pack032_building_assessment_010 | `bea_audit_trail`, `bea_compliance_records`, `bea_tenant_reports`, views, indexes, RLS policies | Audit trail with SHA-256 provenance, regulatory compliance tracking (EPBD/MEES/BPS), tenant energy reports, materialized views for portfolio dashboards, performance indexes, and row-level security policies |

**Table Prefix:** `bea_` (Building Energy Assessment)

**Row-Level Security (RLS):**
- All tables have `building_id` column for building-level access control
- RLS policies enforce that users can only see data for buildings assigned to their role
- Portfolio managers have read access across all buildings in their portfolio
- Tenants have read-only access to their own consumption data
- Admin role bypasses RLS for cross-portfolio reporting

**Indexes:**
- Composite indexes on `(building_id, created_at)` for time-series queries
- GIN indexes on JSONB columns for flexible metadata storage (construction details, system parameters)
- Partial indexes on `status` columns for active-record filtering
- B-tree indexes on `zone_id`, `system_id`, `epc_id`, `assessment_id` for foreign key joins
- Spatial index on building coordinates for geographic portfolio queries

---

## 13. File Structure

```
packs/energy-efficiency/PACK-032-building-energy-assessment/
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
      commercial_office.yaml
      retail_building.yaml
      hotel_hospitality.yaml
      healthcare_facility.yaml
      education_building.yaml
      residential_multifamily.yaml
      mixed_use_development.yaml
      public_sector_building.yaml
  engines/
    __init__.py
    building_envelope_engine.py
    epc_rating_engine.py
    hvac_assessment_engine.py
    domestic_hot_water_engine.py
    lighting_assessment_engine.py
    renewable_integration_engine.py
    building_benchmark_engine.py
    retrofit_analysis_engine.py
    indoor_environment_engine.py
    whole_life_carbon_engine.py
  workflows/
    __init__.py
    initial_building_assessment_workflow.py
    epc_generation_workflow.py
    retrofit_planning_workflow.py
    continuous_building_monitoring_workflow.py
    certification_assessment_workflow.py
    tenant_engagement_workflow.py
    regulatory_compliance_workflow.py
    nzeb_readiness_workflow.py
  templates/
    __init__.py
    epc_report.py
    dec_report.py
    building_assessment_report.py
    retrofit_recommendation_report.py
    building_benchmark_report.py
    certification_scorecard.py
    tenant_energy_report.py
    building_dashboard.py
    regulatory_compliance_report.py
    whole_life_carbon_report.py
  integrations/
    __init__.py
    building_assessment_orchestrator.py
    mrv_building_bridge.py
    data_building_bridge.py
    epbd_compliance_bridge.py
    bms_integration_bridge.py
    weather_data_bridge.py
    certification_bridge.py
    grid_carbon_bridge.py
    property_registry_bridge.py
    health_check.py
    setup_wizard.py
    crrem_pathway_bridge.py
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_building_envelope_engine.py
    test_epc_rating_engine.py
    test_hvac_assessment_engine.py
    test_domestic_hot_water_engine.py
    test_lighting_assessment_engine.py
    test_renewable_integration_engine.py
    test_building_benchmark_engine.py
    test_retrofit_analysis_engine.py
    test_indoor_environment_engine.py
    test_whole_life_carbon_engine.py
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
| Workflow Tests | >85% | All 8 workflows with synthetic building data |
| Template Tests | 100% | All 10 templates in 3+ formats (MD, HTML, JSON, PDF where applicable) |
| Integration Tests | >80% | All 12 integrations with mock agents, BMS simulators, and meter data |
| E2E Tests | Core happy path | Full pipeline from building setup to comprehensive assessment report |
| U-value Tests | 100% | All construction types validated against EN ISO 6946 worked examples |
| EPC Tests | 100% | EPC rating validated against national reference calculation tools (SAP, GEG, DPE) |
| HVAC Tests | >90% | Heat pump SCOP, chiller SEER, boiler AFUE with known engineering reference values |
| DHW Tests | 100% | DHW demand, solar thermal f-chart, legionella cycle with reference calculations |
| Lighting Tests | 100% | LENI calculation, daylight factor, controls credits with EN 15193 worked examples |
| PV Yield Tests | >90% | PV yield validated against PVGIS for 20+ European locations |
| Benchmark Tests | 100% | EUI benchmarks, Energy Star, CRREM pathway for all 8 building types |
| Retrofit Tests | >90% | Measure savings, interaction matrix, NPV/IRR with known-value financial validation |
| IEQ Tests | 100% | PMV/PPD against ISO 7730 tables, adaptive comfort, overheating TM59 |
| WLC Tests | 100% | EN 15978 lifecycle stages, embodied carbon against ICE database values |
| Preset Tests | 100% | All 8 building-type presets with representative building scenarios |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 800+ tests (60-80 per engine, 40-50 integration, 20-30 E2E)

**Known-Value Validation Sets:**
- 200 U-value calculations validated against EN ISO 6946 Annex A worked examples and CIBSE Guide A
- 50 thermal bridge psi-values validated against BRE accredited construction details
- 100 EPC calculations cross-validated against SAP 10.2 and DIN V 18599 reference outputs
- 50 heat pump SCOP calculations validated against EN 14825 Eurovent certified data
- 30 LENI calculations validated against EN 15193-1 worked examples
- 50 PV yield estimations validated against PVGIS database for European locations
- 20 PMV/PPD calculations validated against ISO 7730 Table D.1
- 30 whole life carbon calculations validated against published case studies
- 25 CRREM pathway assessments validated against CRREM online tool

---

## 15. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-20 |
| Phase 2 | Engine implementation (10 engines) | 2026-03-21 to 2026-03-23 |
| Phase 3 | Workflow implementation (8 workflows) | 2026-03-23 to 2026-03-24 |
| Phase 4 | Template implementation (10 templates) | 2026-03-24 to 2026-03-25 |
| Phase 5 | Integration implementation (12 integrations) | 2026-03-25 to 2026-03-26 |
| Phase 6 | Test suite (800+ tests) | 2026-03-26 to 2026-03-28 |
| Phase 7 | Database migrations (V191-V200) | 2026-03-28 |
| Phase 8 | Documentation & Release | 2026-03-29 |

---

## 16. Appendix: U-Value Calculation Worked Examples

### Example 1: Solid Brick Wall (Victorian, uninsulated)

```
Layer 1: External surface resistance    R_se = 0.04 m2.K/W
Layer 2: 215mm solid brick              R = 0.215 / 0.77 = 0.279 m2.K/W
Layer 3: 13mm dense plaster             R = 0.013 / 0.57 = 0.023 m2.K/W
Layer 4: Internal surface resistance    R_si = 0.13 m2.K/W

R_total = 0.04 + 0.279 + 0.023 + 0.13 = 0.472 m2.K/W
U = 1 / 0.472 = 2.12 W/m2.K

Conclusion: Very poor thermal performance. Current Part L (UK) requires U <= 0.30 W/m2.K for new walls.
```

### Example 2: Cavity Wall with Insulation (Post-2006)

```
Layer 1: External surface resistance    R_se = 0.04 m2.K/W
Layer 2: 102.5mm facing brick           R = 0.1025 / 0.77 = 0.133 m2.K/W
Layer 3: 100mm cavity (fully filled, mineral wool) R = 100mm / 0.038 = 2.632 m2.K/W
Layer 4: 100mm lightweight block         R = 0.100 / 0.19 = 0.526 m2.K/W
Layer 5: 13mm plasterboard on dabs      R = 0.013 / 0.21 = 0.062 m2.K/W
Layer 6: Internal surface resistance    R_si = 0.13 m2.K/W

R_total = 0.04 + 0.133 + 2.632 + 0.526 + 0.062 + 0.13 = 3.523 m2.K/W
U = 1 / 3.523 = 0.284 W/m2.K

Conclusion: Meets Part L 2021 limiting U-value of 0.30 W/m2.K.
```

### Example 3: Externally Insulated Solid Wall (Retrofit)

```
Layer 1: External surface resistance    R_se = 0.04 m2.K/W
Layer 2: 10mm render on mesh            R = 0.010 / 0.57 = 0.018 m2.K/W
Layer 3: 120mm EPS insulation (lambda 0.032) R = 0.120 / 0.032 = 3.750 m2.K/W
Layer 4: 215mm solid brick              R = 0.215 / 0.77 = 0.279 m2.K/W
Layer 5: 13mm dense plaster             R = 0.013 / 0.57 = 0.023 m2.K/W
Layer 6: Internal surface resistance    R_si = 0.13 m2.K/W

R_total = 0.04 + 0.018 + 3.750 + 0.279 + 0.023 + 0.13 = 4.240 m2.K/W
U = 1 / 4.240 = 0.236 W/m2.K

Improvement: From U = 2.12 to U = 0.236 (89% reduction in fabric heat loss through this element)
```

### Example 4: Window U-Value (Double Glazed, Argon, Low-E)

```
Glazing: Ug = 1.1 W/m2.K (argon filled, low-e coating)
Frame: Uf = 1.6 W/m2.K (PVC-U, multi-chamber)
Spacer: psi_g = 0.06 W/m.K (warm-edge spacer)

Window dimensions: 1200mm x 1500mm
Frame width: 70mm all around

Ag = (1.200 - 2*0.070) * (1.500 - 2*0.070) = 1.060 * 1.360 = 1.442 m2
Af = (1.200 * 1.500) - 1.442 = 1.800 - 1.442 = 0.358 m2
Lg = 2 * (1.060 + 1.360) = 4.840 m

Uw = (1.442 * 1.1 + 0.358 * 1.6 + 4.840 * 0.06) / (1.442 + 0.358)
Uw = (1.586 + 0.573 + 0.290) / 1.800
Uw = 2.449 / 1.800 = 1.36 W/m2.K

g-value: 0.60 (with low-e coating)
Light transmittance: 0.75
```

---

## 17. Appendix: EPC Rating Methodology

### EPC Rating Scale (Indicative, Non-Residential, Primary Energy)

| Rating | Range (kWh/m2/yr) | Colour | Description |
|--------|-------------------|--------|-------------|
| A | 0 - 50 | Dark Green | Nearly zero-energy / zero-emission building |
| B | 51 - 100 | Green | Very high performance |
| C | 101 - 150 | Yellow-Green | High performance |
| D | 151 - 200 | Yellow | Average performance |
| E | 201 - 250 | Orange | Below average |
| F | 251 - 300 | Dark Orange | Poor performance |
| G | >300 | Red | Worst performing |

Note: Exact thresholds are set by each Member State. Above values are indicative for offices.

### EPC Rating Scale (UK SAP, Residential)

| SAP Rating | Band | Typical Building |
|-----------|------|------------------|
| 92-100+ | A | New build, heat pump, solar PV, high insulation |
| 81-91 | B | New build, condensing boiler, high insulation |
| 69-80 | C | New build (pre-2013), or retrofit with insulation |
| 55-68 | D | 1990s build, or retrofit with cavity fill |
| 39-54 | E | 1960s-1980s, some insulation |
| 21-38 | F | Pre-1960s, limited insulation |
| 1-20 | G | Pre-1919, no insulation, poor heating |

---

## 18. Appendix: CRREM Pathway Data

### CRREM 1.5C Pathway -- Office (Selected Countries, kgCO2/m2/yr)

| Year | UK | Germany | France | Netherlands | Italy | Spain |
|------|-----|---------|--------|-------------|-------|-------|
| 2025 | 42 | 48 | 22 | 45 | 38 | 35 |
| 2030 | 28 | 32 | 15 | 30 | 25 | 23 |
| 2035 | 18 | 20 | 10 | 19 | 16 | 15 |
| 2040 | 10 | 12 | 6 | 11 | 9 | 8 |
| 2045 | 5 | 6 | 3 | 5 | 4 | 4 |
| 2050 | 0 | 0 | 0 | 0 | 0 | 0 |

Note: Values are illustrative approximations. Actual CRREM pathway values are updated annually and vary by exact building sub-type. France has lower intensity due to nuclear-dominated grid.

### CRREM 1.5C Pathway -- Residential (Selected Countries, kgCO2/m2/yr)

| Year | UK | Germany | France | Netherlands | Italy | Spain |
|------|-----|---------|--------|-------------|-------|-------|
| 2025 | 32 | 38 | 15 | 35 | 28 | 25 |
| 2030 | 20 | 24 | 10 | 22 | 18 | 16 |
| 2035 | 12 | 15 | 6 | 13 | 11 | 10 |
| 2040 | 6 | 8 | 3 | 7 | 5 | 5 |
| 2045 | 2 | 3 | 1 | 3 | 2 | 2 |
| 2050 | 0 | 0 | 0 | 0 | 0 | 0 |

---

## 19. Appendix: Retrofit Measure Library Summary (60+ Measures)

| Category | Measures | ID Range | Typical Savings Range | Typical Payback Range |
|----------|---------|----------|----------------------|----------------------|
| Envelope | External wall insulation, internal wall insulation, cavity fill, loft insulation, flat roof insulation, floor insulation (suspended/solid), glazing upgrade, secondary glazing, draught-proofing, air tightness, cool roof, green roof, external shading, thermal bridge remediation | 1-15 | 5-35% heating | 1-30 years |
| HVAC | Boiler replacement, ASHP, GSHP, district heat, heating controls, weather compensation, MVHR, DCV, VSD fans, chiller replacement, VRF, free cooling, cooling tower optimization, pipework insulation, efficient pumps, BMS upgrade, destratification, refrigerant change, night purge, radiant panels | 16-35 | 5-65% system-specific | 1-15 years |
| Lighting | LED replacement (fluorescent, halogen, high-bay), occupancy sensing, daylight dimming, task lighting, external LED, car park LED, emergency LED, constant illuminance | 36-45 | 15-80% lighting | 0.5-7 years |
| Renewables | Rooftop PV, BIPV facade, flat plate solar thermal, evacuated tube solar thermal, battery storage, CHP, small wind, 5GDHC connection | 46-53 | Generation / fuel substitution | 5-20 years |
| Controls & DR | Smart metering, power factor correction, voltage optimization, thermal storage, demand response, BEMS analytics, tenant engagement | 54-60 | 2-15% total | 1-7 years |

---

## 20. Appendix: EN 15978 Lifecycle Stage Definitions

| Module | Stage | Description | Typical Share of WLC |
|--------|-------|-------------|---------------------|
| A1 | Raw material supply | Extraction and processing of raw materials | 30-50% of embodied |
| A2 | Transport to factory | Transport of raw materials to manufacturing | 2-5% of embodied |
| A3 | Manufacturing | Manufacturing of construction products | 20-40% of embodied |
| A4 | Transport to site | Transport of products from factory to construction site | 1-3% of embodied |
| A5 | Construction/installation | On-site construction activities, waste, temporary works | 3-8% of embodied |
| B1 | Use | Emissions from use phase (e.g., carbonation of concrete) | <1% of WLC |
| B2 | Maintenance | Regular maintenance activities | 1-3% of WLC |
| B3 | Repair | Repair of building components | 1-2% of WLC |
| B4 | Replacement | Replacement of building components over building life | 5-15% of WLC |
| B5 | Refurbishment | Major refurbishment activities during building life | 2-8% of WLC |
| B6 | Operational energy use | Energy consumption during building operation | 30-70% of WLC |
| B7 | Operational water use | Water consumption during building operation | 1-3% of WLC |
| C1 | Deconstruction/demolition | Dismantling of building at end of life | 1-2% of WLC |
| C2 | Transport to disposal | Transport of waste materials from site | <1% of WLC |
| C3 | Waste processing | Sorting, shredding, treatment of waste | 1-2% of WLC |
| C4 | Disposal | Landfill or incineration of residual waste | 1-3% of WLC |
| D | Beyond system boundary | Benefits from reuse, recovery, recycling of materials | -5% to -15% (credit) |

---

## 21. Appendix: Thermal Comfort PMV/PPD Reference

### ISO 7730 Table: PMV vs. PPD

| PMV | PPD (%) | Sensation |
|-----|---------|-----------|
| -3 | 100 | Cold |
| -2 | 77 | Cool |
| -1 | 26 | Slightly cool |
| -0.5 | 10 | Slightly cool |
| 0 | 5 | Neutral |
| +0.5 | 10 | Slightly warm |
| +1 | 26 | Slightly warm |
| +2 | 77 | Warm |
| +3 | 100 | Hot |

### Metabolic Rates (ISO 7730)

| Activity | Metabolic Rate (met) | W/m2 |
|----------|---------------------|------|
| Seated, relaxed | 1.0 | 58 |
| Seated, office work | 1.2 | 70 |
| Standing, light work | 1.6 | 93 |
| Standing, medium work | 2.0 | 117 |
| Walking (4 km/h) | 2.6 | 151 |
| Light machine work | 2.0 | 117 |
| Heavy work | 3.0-4.0 | 175-233 |

### Clothing Insulation (ISO 7730)

| Clothing Ensemble | Insulation (clo) | m2.K/W |
|-------------------|-----------------|--------|
| Naked | 0 | 0 |
| Light summer clothing (shorts, short-sleeve shirt) | 0.3 | 0.047 |
| Typical summer office (trousers, short-sleeve shirt) | 0.5 | 0.078 |
| Typical winter office (suit, long-sleeve shirt) | 1.0 | 0.155 |
| Heavy winter clothing (overcoat, hat, gloves) | 1.5 | 0.233 |

---

## 22. User Stories & Acceptance Criteria

### US-001: Building Envelope Assessment

```
As a building energy assessor,
I want to calculate U-values for all building envelope elements,
So that I can accurately determine the fabric heat loss for an EPC assessment.
```

**Acceptance Criteria:**

```
GIVEN a building with known construction layers (material, thickness, conductivity),
WHEN I run the building envelope engine,
THEN it SHALL calculate U-values per EN ISO 6946 within 1% of reference calculation,
  AND apply thermal bridge correction using psi-values or default y-value,
  AND calculate area-weighted mean U-value for the building,
  AND identify elements exceeding current regulatory U-value limits,
  AND complete the calculation in < 30 seconds for 50 elements.

GIVEN a building where construction layers are unknown,
WHEN I enter the building age and type,
THEN it SHALL offer age-based default U-values from TABULA/EPISCOPE typology,
  AND flag these as "default values -- survey recommended",
  AND allow manual override with actual construction data.
```

### US-002: EPC Generation

```
As a building energy assessor,
I want to generate an Energy Performance Certificate for a commercial building,
So that I can issue a legally compliant EPC for sale, rental, or renovation.
```

**Acceptance Criteria:**

```
GIVEN a building with complete envelope, HVAC, lighting, and DHW data,
  AND a selected national EPC methodology (SAP, GEG, DPE, etc.),
WHEN I run the EPC generation workflow,
THEN it SHALL calculate primary energy (kWh/m2/yr) per the selected methodology,
  AND calculate CO2 emissions (kgCO2/m2/yr),
  AND determine the EPC rating (A-G),
  AND generate improvement recommendations ranked by cost-effectiveness,
  AND produce a certificate report in PDF format meeting national layout requirements,
  AND complete the workflow in < 30 minutes.

GIVEN an EPC where the calculated rating is within 5% of a rating boundary,
WHEN the result is presented,
THEN it SHALL flag the proximity to the boundary,
  AND identify the single measure most likely to achieve the next rating band,
  AND indicate the rating sensitivity to input assumptions.
```

### US-003: HVAC System Assessment

```
As a facilities manager,
I want to assess the seasonal efficiency of my building's HVAC systems,
So that I can identify underperforming systems and plan upgrades.
```

**Acceptance Criteria:**

```
GIVEN a building with heating (gas boiler and/or heat pump), cooling (chiller/splits), and ventilation (AHU with heat recovery),
WHEN I run the HVAC assessment engine,
THEN it SHALL calculate heating seasonal efficiency (AFUE for boiler, SCOP for heat pump),
  AND calculate cooling SEER per EN 14825 part-load methodology,
  AND calculate heat pump SPF1-SPF4 including supplementary heater contribution,
  AND assess ventilation heat recovery effectiveness and SFP,
  AND classify building controls per EN 15232 (BACS Class A-D),
  AND track refrigerant inventory with F-gas compliance status.
```

### US-004: Retrofit Planning

```
As a portfolio manager,
I want to develop a staged retrofit plan for an underperforming building,
So that I can achieve nZEB compliance while optimizing financial return.
```

**Acceptance Criteria:**

```
GIVEN a building currently rated EPC E (primary energy 230 kWh/m2/yr),
  AND a target of nZEB (primary energy < 60 kWh/m2/yr),
WHEN I run the retrofit planning workflow,
THEN it SHALL identify applicable measures from the 60+ measure library,
  AND model measure interactions (envelope-HVAC, lighting-HVAC),
  AND calculate interaction-adjusted savings (not simple sum of individual savings),
  AND produce 3-5 retrofit packages at different investment levels,
  AND produce a staged roadmap achieving nZEB in 3-5 stages,
  AND calculate NPV, IRR, and payback for each package,
  AND project year-by-year EPC rating improvement per stage.
```

### US-005: CRREM Pathway Assessment

```
As a real estate investor,
I want to know when my building will become a "stranded asset" under CRREM,
So that I can plan retrofit investment to avoid stranding.
```

**Acceptance Criteria:**

```
GIVEN a building with known type, country, and current carbon intensity (kgCO2/m2/yr),
WHEN I run the CRREM pathway assessment,
THEN it SHALL look up the correct CRREM pathway (1.5C and 2.0C) for the building type and country,
  AND calculate the stranding year under business-as-usual (including grid decarbonization),
  AND calculate the stranding year post-retrofit (for each retrofit package),
  AND produce a chart showing building trajectory vs. CRREM pathway over 30 years,
  AND flag buildings stranding before 2030 as "critical",
  AND flag buildings stranding 2030-2040 as "at risk".

GIVEN a portfolio of 100+ buildings,
WHEN I run the portfolio CRREM analysis,
THEN it SHALL calculate stranding year for each building,
  AND rank buildings by stranding urgency,
  AND calculate aggregate portfolio carbon intensity trajectory,
  AND identify the top 10 buildings requiring priority intervention.
```

### US-006: Whole Life Carbon Assessment

```
As an architect designing a new building,
I want to calculate whole life carbon per EN 15978,
So that I can demonstrate compliance with RIBA/LETI targets and optimize material choices.
```

**Acceptance Criteria:**

```
GIVEN a building design with bill of materials (quantities and materials),
  AND operational energy assessment (from engines 1-7),
  AND a 60-year study period,
WHEN I run the whole life carbon engine,
THEN it SHALL calculate embodied carbon (A1-A5) using EPD or ICE database factors,
  AND calculate use stage carbon (B2-B5 maintenance/replacement, B6 operational energy),
  AND project B6 operational carbon with grid decarbonization over 60 years,
  AND calculate end-of-life carbon (C1-C4),
  AND calculate Module D credits for recyclable materials,
  AND report total WLC in kgCO2e/m2,
  AND compare against RIBA 2030 and LETI targets,
  AND identify top 5 material substitution opportunities with carbon saving.
```

### US-007: Indoor Environment Quality

```
As a building services engineer,
I want to assess indoor environmental quality,
So that I can ensure occupant comfort while optimizing energy consumption.
```

**Acceptance Criteria:**

```
GIVEN a building with zone-level temperature, humidity, and ventilation data,
WHEN I run the indoor environment engine,
THEN it SHALL calculate PMV and PPD per ISO 7730 for each zone,
  AND classify thermal comfort per ISO 7730 Category A/B/C,
  AND assess ventilation adequacy per EN 16798-1 Category I-IV,
  AND estimate CO2 levels based on ventilation rate and occupancy density,
  AND assess overheating risk per CIBSE TM59 Criteria A and B,
  AND calculate daylight factor per EN 17037 for perimeter zones,
  AND produce an integrated IEQ score per zone.
```

### US-008: Green Building Certification

```
As a sustainability consultant,
I want to assess my building's LEED and BREEAM scores simultaneously,
So that I can advise the client on the most achievable certification.
```

**Acceptance Criteria:**

```
GIVEN a building with energy, IEQ, materials, and water assessment data,
WHEN I run the certification assessment workflow for LEED v4.1 and BREEAM International,
THEN it SHALL estimate LEED EA credits (energy) within 2 points of official assessment,
  AND estimate BREEAM Ene credits within 1 credit of official assessment,
  AND produce a credit-by-credit scorecard for each scheme,
  AND predict certification level (e.g., LEED Gold, BREEAM Excellent),
  AND identify the gap to the next certification level with specific improvement actions,
  AND compare the effort/cost of achieving each certification.
```

### US-009: Regulatory Compliance

```
As a property manager with buildings across 5 EU countries,
I want to track all building energy regulatory obligations in one place,
So that I can ensure timely compliance and avoid penalties.
```

**Acceptance Criteria:**

```
GIVEN a portfolio of buildings across UK, Germany, France, Netherlands, and Italy,
WHEN I configure the regulatory compliance workflow,
THEN it SHALL identify all applicable regulations per jurisdiction,
  AND check EPC validity and renewal dates for every building,
  AND check MEES compliance (minimum EPC for rental) per jurisdiction,
  AND track EPBD MEPS (minimum energy performance standards) timeline per building,
  AND check F-gas compliance for all buildings with refrigerant systems,
  AND generate a unified compliance calendar with RAG status,
  AND send advance warnings at 12, 6, 3, and 1 month before each deadline.
```

### US-010: Tenant Energy Reporting

```
As a tenant in a multi-let commercial building,
I want to receive a monthly energy report for my demise,
So that I can monitor my consumption and reduce costs.
```

**Acceptance Criteria:**

```
GIVEN a building with sub-metering per tenant and common area metering,
WHEN the tenant engagement workflow runs at month-end,
THEN it SHALL calculate my metered energy consumption by carrier,
  AND allocate my share of common area energy (by floor area or metered proportion),
  AND calculate my total energy cost (metered + allocated),
  AND compare my EUI against the building average (anonymized peer comparison),
  AND compare my consumption against previous month and same month last year,
  AND provide 3-5 specific recommendations for reducing consumption,
  AND deliver the report via email in PDF format by the 5th of the following month.
```

---

## 23. Appendix: Glossary

| Term | Definition |
|------|-----------|
| **ACH** | Air Changes per Hour -- volume of air entering a space per hour divided by the room volume |
| **AFUE** | Annual Fuel Utilization Efficiency -- seasonal efficiency of a combustion heating system |
| **ASHRAE** | American Society of Heating, Refrigerating and Air-Conditioning Engineers |
| **BACS** | Building Automation and Control Systems -- per EN 15232 |
| **BER** | Building Energy Rating -- Irish EPC equivalent |
| **BIPV** | Building-Integrated Photovoltaics -- PV integrated into building envelope elements |
| **BMS** | Building Management System -- centralized building controls and monitoring |
| **BPS** | Building Performance Standards -- minimum energy/carbon performance requirements |
| **BREEAM** | Building Research Establishment Environmental Assessment Method -- UK green building certification |
| **CDD** | Cooling Degree Days -- measure of cooling demand based on outdoor temperature |
| **CHP** | Combined Heat and Power -- simultaneous generation of heat and electricity |
| **CIBSE** | Chartered Institution of Building Services Engineers |
| **CRREM** | Carbon Risk Real Estate Monitor -- science-based building decarbonization pathways |
| **DEC** | Display Energy Certificate -- operational rating based on actual metered energy (mandatory for UK public buildings) |
| **DGNB** | German Sustainable Building Council certification |
| **DHW** | Domestic Hot Water -- hot water for washing, bathing, cleaning |
| **DPE** | Diagnostic de Performance Energetique -- French EPC methodology |
| **EER** | Energy Efficiency Ratio -- cooling output per unit electrical input at full load |
| **EnerPHit** | Passive House retrofit standard (less stringent than new-build Passive House) |
| **EPBD** | Energy Performance of Buildings Directive -- EU Directive 2024/1275 |
| **EPC** | Energy Performance Certificate -- building energy rating A-G |
| **EPD** | Environmental Product Declaration -- standardized environmental data for construction products |
| **EUI** | Energy Use Intensity -- energy consumption per unit floor area (kWh/m2/yr) |
| **FCU** | Fan Coil Unit -- HVAC terminal unit with fan and heating/cooling coil |
| **GEG** | Gebaudeenergiegesetz -- German Building Energy Act |
| **GIA** | Gross Internal Area -- total floor area measured to internal face of perimeter walls |
| **GSHP** | Ground Source Heat Pump |
| **g-value** | Solar heat gain coefficient of glazing (fraction of solar radiation transmitted through glass) |
| **HDD** | Heating Degree Days -- measure of heating demand based on outdoor temperature |
| **ICE** | Inventory of Carbon and Energy -- embodied carbon database (University of Bath) |
| **IEQ** | Indoor Environmental Quality -- encompasses thermal comfort, air quality, lighting, acoustics |
| **LEED** | Leadership in Energy and Environmental Design -- US green building certification |
| **LENI** | Lighting Energy Numeric Indicator -- annual lighting energy per unit area (kWh/m2/yr) per EN 15193 |
| **LETI** | London Energy Transformation Initiative -- whole life carbon targets |
| **LPD** | Lighting Power Density -- installed lighting power per unit area (W/m2) |
| **MEES** | Minimum Energy Efficiency Standards -- minimum EPC rating for rental properties |
| **MEPS** | Minimum Energy Performance Standards -- building performance thresholds per EPBD |
| **MVHR** | Mechanical Ventilation with Heat Recovery |
| **NIA** | Net Internal Area -- usable floor area excluding walls, columns, stairs, lifts |
| **nZEB** | Nearly Zero-Energy Building -- per EPBD definition |
| **PMV** | Predicted Mean Vote -- thermal comfort index (-3 cold to +3 hot) per ISO 7730 |
| **PPD** | Predicted Percentage Dissatisfied -- percentage of occupants thermally uncomfortable |
| **Psi-value** | Linear thermal transmittance of a thermal bridge (W/m.K) per EN ISO 10211 |
| **SAP** | Standard Assessment Procedure -- UK domestic EPC methodology |
| **SCOP** | Seasonal Coefficient of Performance -- heat pump annual average efficiency per EN 14825 |
| **SEER** | Seasonal Energy Efficiency Ratio -- cooling system annual average efficiency per EN 14825 |
| **SFP** | Specific Fan Power -- fan electrical power per unit airflow (W/(L/s)) |
| **SPF** | Seasonal Performance Factor -- heat pump overall seasonal efficiency including distribution |
| **TFA** | Treated Floor Area -- floor area of heated/conditioned spaces |
| **TMY** | Typical Meteorological Year -- representative annual weather dataset for a location |
| **U-value** | Thermal transmittance -- rate of heat flow through a building element (W/m2.K) |
| **VRF** | Variable Refrigerant Flow -- multi-split HVAC system with variable refrigerant volume |
| **ZEB** | Zero-Emission Building -- per EPBD recast definition (zero on-site fossil fuel combustion) |

---

## 24. Appendix: Future Roadmap (PACK-033 through PACK-036)

| Pack | Name | Category | Focus |
|------|------|----------|-------|
| PACK-033 | Smart Building Energy Pack | Energy Efficiency | AI/ML-driven building energy optimization: predictive HVAC control, fault detection and diagnostics (FDD), digital twin integration, occupancy-based optimization, weather-predictive control, reinforcement learning for HVAC setpoints, anomaly detection with explainability |
| PACK-034 | District Energy Assessment Pack | Energy Efficiency | District heating and cooling network assessment: network hydraulic modelling, supply/return temperature optimization, heat source diversification (waste heat, geothermal, solar thermal), 4th/5th generation district heating (4GDH/5GDHC), prosumer integration, thermal storage |
| PACK-035 | Transport Energy Assessment Pack | Energy Efficiency | Transport and fleet energy audit per EN 16247-4: fleet electrification roadmap, EV charging infrastructure sizing, route optimization for energy efficiency, driver behaviour analytics, modal shift analysis, Scope 3 transport emissions |
| PACK-036 | Water-Energy Nexus Pack | Energy Efficiency | Water-energy nexus assessment: building water consumption audit, water heating energy optimization, rainwater/greywater recycling energy balance, cooling tower water efficiency, legionella compliance with energy optimization, water-related carbon footprint |

---

## 25. Appendix: Building Energy Flow Diagram (Typical Office)

```
ENERGY INPUTS                    END USES                         LOSSES
=============                    =========                        ======

Grid Electricity --+----> Cooling (25-35%) ---------+---> Useful cooling (18%) + Rejection (12%)
(60-80% of total)  |                                |
                   +----> Lighting (20-30%) --------+---> Useful light (15%) + Heat gain (10%)
                   |                                |
                   +----> Small Power (15-25%) -----+---> IT/equipment (12%) + Heat gain (8%)
                   |                                |
                   +----> Fans/Pumps (5-10%) -------+---> Air movement (4%) + Motor heat (4%)
                   |                                |
                   +----> Lifts (2-5%) -------------+---> Transport (1%) + Standby (2%)

Natural Gas -------+----> Heating (60-80% of gas) --+---> Useful heat (55%) + Flue loss (8%)
(15-30% of total)  |                                |                       + Distribution (5%)
                   +----> DHW (20-40% of gas) ------+---> Useful DHW (12%) + Storage loss (3%)
                                                                           + Distribution (3%)

BUILDING SHELL LOSSES (from useful heat):
  Walls:           15-25% of heating need
  Roof:            10-15% of heating need
  Windows:         15-25% of heating need
  Floor:           5-10% of heating need
  Air infiltration: 15-25% of heating need
  Ventilation:     15-30% of heating need

INTERNAL HEAT GAINS (offset heating, increase cooling):
  Occupants:       5-10 W/m2
  Lighting:        5-15 W/m2 (installed), contributes as heat gain
  Equipment:       10-25 W/m2 (office IT), contributes as heat gain
  Solar gains:     Variable (orientation, glazing, shading dependent)
```

---

**Approval Signatures:**

- Product Manager: ___________________
- Engineering Lead: ___________________
- CEO: ___________________

---

*Document generated by GL-ProductManager | GreenLang Platform v1.0.0 | 2026-03-20*
