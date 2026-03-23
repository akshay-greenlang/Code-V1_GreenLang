# PRD-PACK-037: Demand Response Pack

**Pack ID:** PACK-037-demand-response
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-23
**Prerequisite:** None (standalone; enhanced with PACK-036 Utility Analysis Pack and PACK-035 Energy Benchmark Pack if present; complemented by PACK-031/032/033 Energy Efficiency Packs)

---

## 1. Executive Summary

### 1.1 Problem Statement

Demand response (DR) represents one of the most financially valuable and grid-critical energy management strategies available to commercial, industrial, and institutional facilities. The EU Electricity Market Design Regulation (EU) 2024/1747, FERC Order 2222 in the US, and numerous national grid codes increasingly mandate or incentivize demand-side flexibility. Yet most organizations fail to capture their full DR potential due to persistent challenges:

1. **Unknown load flexibility**: Facilities typically have 10-30% of their peak load that can be curtailed, shifted, or shed during DR events. However, most facility managers cannot quantify their flexible load portfolio because loads are not categorized by criticality, sheddability, or shift potential. Without systematic load disaggregation and flexibility assessment, organizations undercommit to DR programs (leaving revenue on the table) or overcommit (risking operational disruption and non-performance penalties).

2. **Complex DR program landscape**: Over 200 DR programs exist across US ISOs/RTOs (PJM, ERCOT, CAISO, ISO-NE, NYISO, MISO, SPP), EU member state aggregator markets, and UK National Grid ESO flexibility services. Programs differ in notification time (day-ahead, 2-hour, 10-minute), event duration (1-8 hours), performance measurement (baseline methodology CBL vs. metering), penalties for non-performance (capacity payment clawback, imbalance charges), and compensation structure (capacity payments $/kW-month, energy payments $/MWh, ancillary service payments). Without automated program matching, facilities enroll in suboptimal programs or miss eligibility windows.

3. **Inaccurate baseline estimation**: DR performance is measured as the difference between a customer baseline load (CBL) and actual load during an event. Baseline methodologies vary by program (PJM uses "High 4 of 5" with same-day adjustment, CAISO uses "10 of 10", NYISO uses average of 5 highest similar days, UK uses "deemed profile"). Errors in baseline estimation directly translate to revenue loss or non-compliance penalties. Most facilities lack the analytical capability to simulate baseline outcomes under different methodologies and optimize their pre-event consumption to maximize baseline.

4. **Suboptimal dispatch strategies**: When a DR event is called, facilities must decide which loads to curtail, in what sequence, and for how long. Naive approaches (shut everything off) cause operational disruption, product quality issues, and occupant comfort complaints. Optimal dispatch requires multi-objective optimization balancing curtailment magnitude (kW), operational impact (criticality scores), comfort constraints (temperature limits, lighting minimums), product quality bounds, and rebound effects (load snapback post-event). Without automated dispatch optimization, facilities achieve 60-70% of theoretical curtailment capacity.

5. **Revenue leakage and penalty risk**: DR revenue streams include capacity payments (monthly $/kW), energy payments (per-event $/MWh), ancillary service payments (regulation, spinning reserve), and avoided demand charges. Performance shortfall triggers penalties: PJM charges 1.2x capacity payment for non-performance, ERCOT applies "clawback" provisions, UK applies capacity market penalties of 1/24th of monthly payment per missed hour. Without automated performance tracking and revenue reconciliation, facilities leave 20-40% of potential DR revenue uncaptured and face unexpected penalty exposure.

6. **Missing carbon and grid impact quantification**: DR events typically occur during peak demand periods when marginal grid emission factors are highest (gas peakers at 0.5-0.8 tCO2/MWh vs. average grid at 0.2-0.4 tCO2/MWh). The carbon benefit of demand response is 2-3x higher per MWh than average grid displacement. Yet most organizations cannot quantify the Scope 2 emission reduction from DR participation for GHG Protocol reporting, SBTi target tracking, or CSRD/ESRS E1 disclosure because they lack marginal emission factor data and event-specific consumption data.

7. **Thermal storage and pre-conditioning ignorance**: Buildings with significant thermal mass (concrete, water tanks, ice storage) can pre-cool or pre-heat before DR events, effectively storing energy in thermal mass to ride through curtailment periods without comfort impact. This strategy can increase effective curtailment capacity by 30-60% while maintaining occupant comfort. However, pre-conditioning requires precise thermal modeling, weather forecasting, and event timing coordination that most facilities cannot perform manually.

8. **Battery and DER coordination gaps**: On-site distributed energy resources (DERs) including battery energy storage systems (BESS), solar PV, backup generators, EV chargers, and combined heat and power (CHP) can be orchestrated to maximize DR performance. Battery dispatch during DR events can deliver sustained curtailment without operational impact. Solar PV can offset load during daytime events. EV charging can be deferred or vehicle-to-building (V2B) can export power. However, DER coordination requires real-time optimization across multiple assets with different state-of-charge, cycling constraints, and degradation profiles.

### 1.2 Solution Overview

PACK-037 is the **Demand Response Pack** -- the seventh pack in the "Energy Efficiency Packs" category. While PACK-031 through PACK-036 focus on energy efficiency (reducing total consumption), PACK-037 focuses on energy flexibility (shifting, shedding, and reshaping consumption in response to grid signals, price signals, and DR program requirements).

The pack provides automated load flexibility assessment across all facility loads, DR program matching against 200+ programs worldwide, customer baseline load (CBL) simulation with 8 baseline methodologies, multi-objective dispatch optimization for DR events, real-time event management with automated load control sequences, DER/battery/thermal storage coordination, performance tracking with revenue reconciliation and penalty avoidance, and carbon impact quantification using marginal emission factors.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete demand response lifecycle from flexibility assessment through settlement verification.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-037 Demand Response Pack |
|-----------|------------------------------|-------------------------------|
| Load flexibility assessment | Manual walkthrough (30% coverage) | Systematic scan of all loads with flexibility scoring (95%+ coverage) |
| Time to DR readiness | 3-6 months | <2 weeks (10-20x faster) |
| Program matching | Manual research of 2-3 programs | Automated matching against 200+ programs |
| Baseline accuracy | Single methodology, unoptimized | 8 methodologies simulated, pre-event optimization |
| Dispatch optimization | Manual load shedding sequence | Multi-objective optimization (kW, comfort, operations) |
| Event response time | 30-60 minutes manual coordination | <5 minutes automated dispatch |
| Revenue capture | 60-70% of potential | 85-95% of potential (25-40% improvement) |
| DER coordination | Manual or none | Automated battery/solar/EV/thermal orchestration |
| Carbon quantification | Not performed | Marginal emission factors, event-specific Scope 2 |
| Performance tracking | Spreadsheet-based | Real-time dashboard with settlement verification |
| Penalty avoidance | Reactive | Predictive non-performance alerts with curtailment verification |
| Audit trail | None | SHA-256 provenance, full calculation lineage |

### 1.4 Demand Response Definition

| Concept | Description |
|---------|-------------|
| **Demand Response (DR)** | Voluntary or mandatory reduction, shift, or reshaping of electricity consumption in response to grid conditions, price signals, or aggregator dispatch |
| **Load Curtailment** | Reducing electricity consumption below baseline during a DR event |
| **Load Shifting** | Moving electricity consumption from peak to off-peak periods |
| **Load Shedding** | Temporarily disconnecting non-critical loads during grid emergencies |
| **Demand Flexibility** | The ability of a load to modify its consumption pattern without unacceptable operational impact |
| **Customer Baseline Load (CBL)** | Estimated counterfactual consumption (what the customer would have consumed without the DR event) |
| **Curtailment Capacity (kW)** | Maximum sustainable load reduction available for DR events |
| **Notification Time** | Lead time between event dispatch and required response (day-ahead to 10 minutes) |
| **Event Duration** | Length of required curtailment (typically 1-8 hours) |

### 1.5 DR Program Types

| # | Program Type | Response Time | Duration | Compensation | Examples |
|---|-------------|---------------|----------|--------------|---------|
| 1 | Economic DR (Day-Ahead) | 18-24 hours | 1-8 hours | $/MWh energy payment | PJM Economic DR, CAISO PDR |
| 2 | Emergency DR | 30 min - 2 hours | 2-6 hours | $/kW-month capacity + $/MWh energy | PJM Emergency DR, NYISO EDRP |
| 3 | Capacity Market | Seasonal commitment | 4-8 hours | $/kW-year capacity | PJM RPM, UK CM, ISO-NE FCM |
| 4 | Ancillary Services | 10 min - 30 min | 15 min - 1 hour | $/MWh regulation/reserve | PJM RegD, ERCOT RRS, CAISO AS |
| 5 | Critical Peak Pricing (CPP) | Day-ahead | 4-6 hours | Rate multiplier (5-10x) | Utility CPP tariffs |
| 6 | Real-Time Pricing (RTP) | Hourly | Continuous | Hourly marginal price | Wholesale-indexed tariffs |
| 7 | Time-of-Use (TOU) Optimization | Pre-scheduled | 2-6 hours peak | Peak/off-peak rate differential | Standard TOU tariffs |
| 8 | Grid Flexibility Services | 30 min - 4 hours | 30 min - 4 hours | $/kW/hr availability | UK Demand Flexibility Service, EU Art 17 flexibility |
| 9 | Behind-the-Meter Optimization | Continuous | Continuous | Demand charge reduction | Demand charge management |
| 10 | Virtual Power Plant (VPP) | Varies | Varies | Aggregated portfolio value | Aggregator-managed portfolios |

### 1.6 Target Users

**Primary:**
- Energy managers seeking to monetize facility load flexibility through DR program participation
- Facility managers responsible for implementing DR event response procedures
- DR aggregators managing portfolios of commercial and industrial customers
- Grid operators and utilities administering DR programs

**Secondary:**
- Corporate sustainability teams quantifying carbon benefits of DR participation
- CFOs evaluating DR revenue streams and ROI on enabling technologies
- Building automation engineers configuring automated DR response sequences
- Portfolio managers rolling out DR across multiple facilities
- ESCOs integrating DR into energy performance contracts
- Microgrid operators coordinating DER dispatch for grid services

### 1.7 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Load flexibility assessment accuracy | Within 10% of measured curtailment | Validated against actual DR event performance |
| DR program matching coverage | >90% of eligible programs identified | Cross-validated against manual program research |
| Baseline estimation accuracy | Within 5% of actual baseline (per program methodology) | Validated against program settlement baselines |
| Dispatch optimization effectiveness | >85% of theoretical curtailment capacity achieved | Measured during actual DR events |
| Event response time (automated) | <5 minutes from notification to load action | Measured from dispatch signal to first load curtailment |
| Revenue capture rate | >85% of maximum potential DR revenue | Compared to theoretical maximum based on actual events |
| Penalty avoidance rate | <5% of events with performance shortfall penalty | Tracked across all enrolled programs |
| Carbon quantification accuracy | Within 10% of verified marginal emission reduction | Cross-validated against marginal emission factor databases |
| Customer NPS | >55 | Net Promoter Score survey |
| System availability during events | >99.5% | Uptime during DR event windows |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EU Electricity Market Design Regulation | Regulation (EU) 2024/1747 | Article 17 mandates demand response participation rights; Article 40 requires aggregator market access |
| EU Energy Efficiency Directive (EED) | Directive (EU) 2023/1791 | Article 11 demand response integration in energy audits; Article 8 efficiency obligation schemes |
| FERC Order 2222 | 18 CFR Parts 35, 271 | Enables DER aggregation participation in US wholesale markets including DR |
| FERC Order 745 | 18 CFR Part 35 | Requires ISOs/RTOs to compensate DR at LMP when cost-effective |
| ISO 50001:2018 | Energy management systems | Demand response as part of energy management planning and operational control |
| OpenADR 2.0b | Open Automated Demand Response | Standard communication protocol for automated DR signal exchange |

### 2.2 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| IEEE 2030.5 | Smart Energy Profile 2.0 | DER communication standard for DR-enabled devices |
| IEC 62746-10-1 | DR communication standard | International DR signal protocol |
| NAESB REQ.18 | DR measurement & verification | North American baseline calculation standards |
| EN 15232-1:2017 | Building automation impact | BACS efficiency classes for automated DR capability |
| IEC 61968/61970 | CIM for energy management | Common Information Model for utility integration |
| ISO 52000-1 | Energy performance of buildings | Flexible energy use in building energy calculations |

### 2.3 Market and Carbon Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 2 emission reduction from DR events using marginal emission factors |
| GHG Protocol Scope 2 Guidance | WRI/WBCSD (2015) | Location-based and market-based accounting for DR carbon benefits |
| SBTi Corporate Framework | SBTi (2024) | DR carbon reductions contributing to science-based targets |
| ESRS E1 Climate Change | EU CSRD (2023) | E1-5 energy consumption disclosure including DR participation |
| WRI/WBCSD Marginal Emission Factors | Cambium, AVERT, Grid Carbon Intensity | Marginal vs. average emission factors for DR carbon quantification |
| PJM Manual 11 | PJM (2024) | Emergency and economic DR settlement procedures |
| CAISO ESDER | CAISO (2024) | Energy storage and distributed energy resources DR participation |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Load flexibility, DR program, baseline, dispatch, event management, DER coordination, performance, revenue, carbon, and reporting engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report, dashboard, and analysis templates |
| Integrations | 12 | Agent, app, data, grid, and system bridges |
| Presets | 8 | Facility-type-specific DR configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `load_flexibility_engine.py` | Systematic assessment of all facility loads for DR flexibility. Categorizes loads by criticality (critical/essential/deferrable/sheddable), curtailment capacity (kW), minimum runtime constraints, ramp rate limits, rebound characteristics, and comfort/operational impact. Builds a load flexibility register covering HVAC (chillers, AHUs, RTUs, VAVs), lighting (zones, dimming capability), plug loads (IT, vending, workstation), motors/drives (pumps, fans, compressors with VSD), process loads (batch vs. continuous), refrigeration (thermal mass ride-through), EV charging (smart vs. dumb, V2B capability), and water heating (thermal storage potential). Calculates total curtailment capacity by notification time (immediate, 10-min, 30-min, 2-hour, day-ahead) and duration (1, 2, 4, 6, 8 hours). |
| 2 | `dr_program_engine.py` | Matches facility flexibility profile to 200+ DR programs worldwide. Database covers US ISOs (PJM, ERCOT, CAISO, ISO-NE, NYISO, MISO, SPP), EU aggregator markets (DE, FR, UK, NL, BE, IT, ES, AT), and direct utility programs. For each eligible program: compensation structure (capacity $/kW-month, energy $/MWh, ancillary $/MWh), performance requirements (minimum curtailment, response time, duration), baseline methodology, penalty structure, enrollment windows, and historical event frequency. Calculates expected annual revenue per program and recommends optimal program portfolio considering stacking rules and exclusivity constraints. |
| 3 | `baseline_engine.py` | Simulates customer baseline load (CBL) using 8 industry-standard methodologies: PJM High 4 of 5 (with same-day symmetric additive adjustment), CAISO 10 of 10, NYISO Average of 5 highest, ERCOT 10CP, UK Deemed Profile, ISO-NE Baseline Type I/II, EU Standard Profile, and Custom regression baseline. For each methodology: calculates projected baseline from historical interval data (15-min or hourly), applies adjustment factors (weather, production, day type), simulates baseline outcomes for past events, identifies optimal pre-event consumption strategy to maximize baseline (within program rules), and quantifies baseline risk (probability of under-performance). Uses Decimal arithmetic for settlement-grade accuracy. |
| 4 | `dispatch_optimizer_engine.py` | Multi-objective optimization engine for DR event dispatch. Given a curtailment target (kW), event duration, and notification time, determines optimal load curtailment sequence. Objective function minimizes: weighted sum of operational disruption (criticality scores), comfort deviation (temperature/lighting below thresholds), product quality risk (process constraint violations), and rebound magnitude (post-event load snapback). Constraints: minimum curtailment target, maximum comfort deviation, critical load protection, minimum equipment runtime, ramp rate limits, and DER availability. Solver uses linear programming (PuLP/scipy) for deterministic, reproducible solutions. Outputs: ordered curtailment sequence, per-load kW reduction, timing offsets, pre-conditioning commands, and DER dispatch schedule. |
| 5 | `event_manager_engine.py` | Real-time DR event lifecycle management from notification through settlement. Phases: (1) Event Registration (receive dispatch signal, parse parameters), (2) Pre-Event Preparation (activate pre-conditioning, verify DER readiness, confirm curtailment plan), (3) Event Execution (issue load control commands, monitor curtailment performance, adjust dispatch in real-time), (4) Event Termination (release curtailed loads, manage rebound, return to normal operations), (5) Post-Event Assessment (calculate actual curtailment, compare to baseline, estimate revenue/penalties). Maintains event log with millisecond timestamps for settlement disputes. Supports OpenADR 2.0b signal parsing and REST API dispatch. |
| 6 | `der_coordinator_engine.py` | Coordinates distributed energy resources for DR events. Asset types: Battery Energy Storage Systems (BESS) with state-of-charge management, depth-of-discharge limits, and cycling optimization; Solar PV with generation forecasting and export/self-consumption optimization; Backup Generators with emissions constraints, runtime limits, and fuel tracking; EV Chargers with smart charging deferral, V2B export, and fleet priority management; Thermal Storage (ice, chilled water, hot water) with charge/discharge optimization; CHP/Cogeneration with heat-led vs. electricity-led dispatch. Calculates combined DER contribution to curtailment capacity. Respects individual asset constraints (SOC limits, cycling counts, emissions permits, fuel reserves). |
| 7 | `performance_tracker_engine.py` | Tracks DR event performance against program requirements. For each event: calculates actual curtailment (kW and MWh) using meter data, compares to baseline (per program methodology), determines compliance (met/exceeded/shortfall), calculates performance ratio (actual/committed), estimates revenue earned, estimates penalties incurred, and generates settlement-ready documentation. Aggregates performance across all events per season: total curtailment delivered, compliance rate, total revenue, total penalties, net DR income. Identifies performance trends and flags deteriorating curtailment capacity. |
| 8 | `revenue_optimizer_engine.py` | Optimizes DR revenue across multiple programs and revenue streams. Revenue streams: capacity payments (monthly/seasonal), energy payments (per-event), ancillary service payments (regulation, reserve), demand charge savings (behind-the-meter), and avoided energy costs (price arbitrage). Calculates: gross revenue per program, penalties and clawbacks, net revenue, enabling technology costs (controls, metering, communication), ROI on DR participation, and marginal value of additional curtailment capacity. Performs what-if analysis: revenue impact of adding loads, upgrading controls, adding battery storage, or changing program enrollment. Uses Decimal arithmetic for financial precision. |
| 9 | `carbon_impact_engine.py` | Calculates carbon impact of DR participation using marginal emission factors. During peak demand, marginal generators are typically gas peakers (0.5-0.8 tCO2/MWh) vs. average grid (0.2-0.4 tCO2/MWh). For each DR event: determines marginal emission factor based on grid region, time-of-day, and season (from Cambium, AVERT, or regional grid carbon databases); calculates Scope 2 emission reduction using marginal factor (vs. average factor for non-DR periods); separates location-based and market-based accounting; quantifies SBTi target contribution; and calculates marginal abatement cost ($/tCO2e avoided). Tracks cumulative carbon benefits of DR program participation for GHG Protocol reporting and CSRD/ESRS E1 disclosure. |
| 10 | `dr_reporting_engine.py` | Aggregates all DR analysis results into configurable reports and dashboards. Dashboard panels: Flexibility Profile (load breakdown by curtailability), Program Portfolio (enrolled programs, commitments, revenue forecast), Event History (timeline of DR events with performance), Revenue Tracker (capacity + energy + ancillary payments vs. penalties), Carbon Impact (marginal emission reductions per event), DER Performance (battery SOC, solar contribution, EV flexibility), Compliance Monitor (program requirement tracking), and Forecast (upcoming event probability, revenue projection). Supports export in MD, HTML, PDF, JSON, and CSV formats. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `flexibility_assessment_workflow.py` | 4: LoadInventory -> FlexibilityScoring -> CurtailmentCapacity -> FlexibilityReport | End-to-end load flexibility assessment from inventory through curtailment capacity quantification |
| 2 | `program_enrollment_workflow.py` | 4: ProgramMatching -> RevenueProjection -> EnrollmentDocumentation -> CommitmentRegistration | DR program selection, revenue analysis, and enrollment preparation |
| 3 | `event_preparation_workflow.py` | 3: EventNotification -> DispatchOptimization -> PreConditioningActivation | Event preparation from signal receipt through pre-event optimization |
| 4 | `event_execution_workflow.py` | 4: LoadCurtailment -> RealTimeMonitoring -> PerformanceVerification -> LoadRestoration | Real-time event management from curtailment through restoration |
| 5 | `settlement_workflow.py` | 3: BaselineCalculation -> PerformanceMeasurement -> RevenueSettlement | Post-event settlement with baseline comparison and revenue calculation |
| 6 | `der_optimization_workflow.py` | 3: AssetInventory -> DispatchStrategy -> CoordinatedResponse | DER coordination for DR events with multi-asset optimization |
| 7 | `reporting_workflow.py` | 3: DataAggregation -> ReportGeneration -> DistributionDelivery | Report and dashboard generation for stakeholders |
| 8 | `full_dr_lifecycle_workflow.py` | 8: FlexibilityAssessment -> ProgramSelection -> Enrollment -> EventPreparation -> EventExecution -> Settlement -> CarbonQuantification -> Reporting | Complete DR lifecycle from assessment through reporting |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `flexibility_profile_report.py` | MD, HTML, PDF, JSON | Load flexibility assessment showing all loads categorized by curtailability, with total curtailment capacity by notification time and duration |
| 2 | `program_analysis_report.py` | MD, HTML, PDF, JSON | DR program comparison with eligibility, revenue projections, penalty risk, and enrollment recommendations |
| 3 | `baseline_analysis_report.py` | MD, HTML, PDF, JSON | CBL methodology comparison showing projected baseline under each methodology with optimization opportunities |
| 4 | `dispatch_plan_report.py` | MD, HTML, PDF, JSON | Load curtailment sequence for DR events with timing, kW reduction, DER dispatch, and pre-conditioning schedule |
| 5 | `event_performance_report.py` | MD, HTML, PDF, JSON | Post-event analysis with actual vs. baseline curtailment, revenue earned, penalties, and lessons learned |
| 6 | `revenue_dashboard.py` | MD, HTML, JSON | Real-time revenue tracking across all programs with capacity, energy, and ancillary payment breakdowns |
| 7 | `carbon_impact_report.py` | MD, HTML, PDF, JSON | Carbon benefits of DR participation using marginal emission factors with SBTi alignment and CSRD reporting |
| 8 | `der_performance_report.py` | MD, HTML, PDF, JSON | DER asset performance during DR events: battery SOC, solar contribution, EV flexibility, thermal storage utilization |
| 9 | `executive_summary_report.py` | MD, HTML, PDF, JSON | 2-4 page C-suite summary with total DR revenue, carbon impact, program compliance, and strategic recommendations |
| 10 | `settlement_verification_report.py` | MD, HTML, PDF, JSON | Settlement-grade documentation for program administrators with baseline calculations, meter data, and performance verification |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 12-phase DAG pipeline: HealthCheck -> Configuration -> LoadInventory -> FlexibilityAssessment -> ProgramMatching -> BaselineSimulation -> DispatchOptimization -> DERCoordination -> EventManagement -> PerformanceTracking -> RevenueReconciliation -> ReportGeneration. Conditional phases for DER coordination (if DERs present) and thermal pre-conditioning (if thermal storage available). Retry with exponential backoff, SHA-256 provenance chain, phase-level caching. |
| 2 | `mrv_bridge.py` | Routes to AGENT-MRV agents for emissions linked to DR curtailment: MRV-009 (Location-Based Scope 2 for electricity curtailment), MRV-010 (Market-Based Scope 2), MRV-013 (Dual Reporting). Provides marginal emission factors for peak-period curtailment to calculate enhanced carbon benefits of DR vs. average grid displacement. |
| 3 | `data_bridge.py` | Routes to AGENT-DATA agents: DATA-002 (Excel/CSV for meter interval data), DATA-001 (PDF for utility bills and program documentation), DATA-003 (ERP/Finance for energy costs and DR revenue), DATA-010 (Data Quality Profiler for meter data validation), DATA-014 (Time Series Gap Filler for interval data gaps), DATA-015 (Cross-Source Reconciliation for meter vs. billing data). |
| 4 | `grid_signal_bridge.py` | External grid signal integration: OpenADR 2.0b signal reception and parsing, ISO/RTO dispatch signal APIs (PJM, ERCOT, CAISO, ISO-NE, NYISO), aggregator platform APIs, utility CPP/RTP price signal reception, grid carbon intensity signals (WattTime, Electricity Maps), and weather forecast services for thermal pre-conditioning. |
| 5 | `bms_control_bridge.py` | Building Management System control integration: read/write capability for HVAC setpoints, lighting levels, equipment schedules, and load control relays. Supports BACnet/IP, Modbus TCP/RTU, OPC-UA, and REST API protocols. Critical for automated DR event execution with load curtailment commands. |
| 6 | `meter_data_bridge.py` | Advanced Metering Infrastructure (AMI) integration for 15-minute or higher resolution interval data. Supports Green Button (ESPI/CMD), IEC 61968, Modbus meter reads, and utility interval data APIs. Real-time meter data during events for performance verification. |
| 7 | `der_asset_bridge.py` | DER asset communication: battery management systems (Modbus, SunSpec, manufacturer APIs), solar PV inverters (SunSpec, SMA, Enphase, SolarEdge APIs), EV charging stations (OCPP 1.6/2.0.1), backup generators (Modbus, manufacturer APIs), and thermal storage controllers. Provides real-time SOC, availability, and dispatch command interface. |
| 8 | `pack036_bridge.py` | PACK-036 Utility Analysis integration: imports utility rate structures, TOU periods, demand charge data, and billing analysis. DR revenue calculations use exact tariff data from PACK-036. Demand charge savings from load curtailment calculated using PACK-036 rate models. |
| 9 | `pack033_bridge.py` | PACK-033 Quick Wins Identifier integration: quick wins related to load controls, scheduling, and demand management feed into DR flexibility assessment. Measures identified by PACK-033 (BMS optimization, scheduling) directly enable DR capability. |
| 10 | `health_check.py` | 20-category system verification covering all 10 engines, 8 workflows, database connectivity, cache status, MRV bridge, DATA bridge, grid signal connectivity, BMS connectivity, meter data freshness, DER asset status, program enrollment status, and authentication/authorization. |
| 11 | `setup_wizard.py` | 9-step guided DR configuration: facility profile (type, area, peak demand, operating hours), load inventory (HVAC, lighting, plug loads, motors, process, EV, DER), utility accounts (rate structure, meter ID, interval data access), DR program preferences (programs of interest, risk tolerance, minimum revenue threshold), grid region (ISO/RTO, utility territory, aggregator), baseline data requirements (historical interval data period), DER assets (battery specs, solar capacity, EV count, thermal storage), BMS connectivity (protocol, endpoints, control capability), and reporting preferences. |
| 12 | `alert_bridge.py` | Alert and notification integration for DR events: event dispatch notifications (SMS, email, push, webhook), pre-event preparation reminders, real-time performance alerts (curtailment shortfall warning), post-event performance summary, enrollment deadline reminders, settlement notifications, and program rule change alerts. Multi-channel delivery with escalation for critical DR events. |

### 3.6 Presets

| # | Preset | Facility Type | Key Characteristics |
|---|--------|--------------|---------------------|
| 1 | `commercial_office.yaml` | Commercial Office | Curtailable: HVAC (30-50% of peak), lighting dimming/zones (10-20%), plug loads (5-10%), EV charging (5-15%). Typical curtailment: 15-35% of peak demand. Comfort constraints: +/- 2C temperature band, 300 lux minimum. Focus on pre-cooling, lighting reduction, and EV deferral. |
| 2 | `manufacturing.yaml` | Manufacturing/Industrial | Curtailable: compressed air (10-20%), motors/drives (15-30%), process scheduling (10-25%), HVAC (5-15%). Typical curtailment: 10-30% of peak demand. Production constraints: batch scheduling flexibility, minimum runtime, product quality. Focus on process load shifting and motor speed reduction. |
| 3 | `retail_grocery.yaml` | Retail/Grocery | Curtailable: HVAC (15-25%), lighting (10-20%), refrigeration (5-15% with thermal ride-through), EV charging (if present). Typical curtailment: 10-25% of peak demand. Constraints: refrigeration temperature limits, customer comfort. Focus on pre-cooling, anti-sweat heater control, and lighting reduction. |
| 4 | `warehouse_cold.yaml` | Warehouse/Cold Storage | Curtailable: refrigeration compressors (20-40% with thermal mass), lighting (10-20%), HVAC (5-10%), dock equipment. Typical curtailment: 15-35% of peak demand. Key advantage: significant thermal storage in product mass. Focus on pre-cooling and compressor cycling. |
| 5 | `healthcare.yaml` | Healthcare | Curtailable: non-clinical HVAC (10-15%), lighting in admin areas (5-10%), EV charging (3-5%), kitchen/laundry scheduling. Typical curtailment: 5-15% of peak demand. Critical constraints: clinical area exclusion, medical equipment protection, infection control ventilation. Conservative DR participation. |
| 6 | `education_campus.yaml` | Education/Campus | Curtailable: HVAC (20-35%), lighting (10-20%), IT labs (5-10%), EV charging (5-15%). Typical curtailment: 15-30% of peak demand. Scheduling advantage: known occupancy patterns, summer/holiday availability. Focus on pre-cooling and scheduling-based curtailment. |
| 7 | `data_center.yaml` | Data Center | Curtailable: cooling (5-15% with raised setpoints), UPS bypass mode (2-5%), lighting (1-2%), EV charging (5-10%). Typical curtailment: 5-15% of IT peak. Key constraint: IT uptime, PUE impact. Focus on temperature setpoint raise within ASHRAE A1 envelope, battery UPS co-optimization. |
| 8 | `microgrid_der.yaml` | Microgrid/DER-Rich | Full DER orchestration: battery (100% of rated capacity), solar (available generation), backup generator (permitted capacity), EV V2B (available fleet), thermal storage. Typical curtailment: 30-80% of peak demand via DER dispatch. Focus on coordinated DER response and islanding capability. |

---

## 4. Engine Specifications

### 4.1 Engine 1: Load Flexibility Engine

**Purpose:** Systematic assessment of all facility loads for demand response flexibility.

**Load Categorization:**

| Criticality Level | Definition | DR Treatment | Examples |
|-------------------|-----------|--------------|---------|
| Critical (Level 0) | Cannot be curtailed under any circumstances | Excluded from DR | Life safety, fire systems, security, medical equipment, data center IT |
| Essential (Level 1) | Curtailment only in grid emergency (30-60 min max) | Emergency DR only | Elevators (one car), emergency lighting, communication systems |
| Deferrable (Level 2) | Can be deferred 1-4 hours without operational impact | Load shifting candidate | EV charging, water heating, batch processes, laundry |
| Sheddable (Level 3) | Can be shed for 2-8 hours with manageable impact | Full DR candidate | Non-critical HVAC zones, decorative lighting, vending, plug loads |
| Flexible (Level 4) | Continuously adjustable with minimal impact | Regulation/ancillary candidate | VSD-controlled motors, dimming lighting, smart EV charging |

**Flexibility Scoring:**

```
Flexibility_Score = (
    curtailment_capacity_kw * 0.30 +        # How much can be curtailed
    response_speed_factor * 0.15 +            # How fast can it respond
    duration_factor * 0.15 +                  # How long can it sustain curtailment
    rebound_factor * 0.10 +                   # Severity of post-event snapback
    comfort_impact_factor * 0.10 +            # Impact on occupant comfort
    operational_impact_factor * 0.10 +        # Impact on operations/production
    automation_factor * 0.10                  # Level of automation available
)
```

**Key Models:**
- `LoadProfile` - Load ID, name, type, rated power kW, average power kW, operating schedule, criticality level, flexibility parameters
- `FlexibilityAssessment` - Load ID, curtailment capacity kW, response time, maximum curtailment duration, rebound characteristics, comfort impact score, operational impact score
- `CurtailmentCapacity` - Total capacity by notification time and duration, capacity by load type, seasonal variation, time-of-day variation
- `FlexibilityRegister` - Complete facility flexibility inventory with per-load assessments

**Non-Functional Requirements:**
- Full facility assessment (500 loads): <15 minutes
- Load profile analysis: <500ms per load
- Reproducibility: bit-perfect (same input = same output, SHA-256 verified)

### 4.2 Engine 2: DR Program Engine

**Purpose:** Match facility flexibility profile to DR programs and optimize program portfolio.

**Program Database (200+ programs):**

| Region | Programs | Key Programs |
|--------|----------|-------------|
| PJM | 15+ | Economic DR, Emergency DR, PRD, Capacity Performance, RegD/RegA |
| ERCOT | 10+ | ERS, LR, RRS, ECRS, 4CP demand response |
| CAISO | 12+ | PDR, RDRR, Base Interruptible Program, CPP, DBP |
| ISO-NE | 8+ | FCM DR, Real-Time DR, Daily DR |
| NYISO | 10+ | EDRP, ICAP/SCR, DADRP, CSR |
| MISO | 6+ | LMR Type I/II, Emergency DR |
| UK | 15+ | Capacity Market, DFS, FFR, STOR, Balancing Mechanism |
| Germany | 12+ | AbLaV, Regelleistung (FCR, aFRR, mFRR), Spitzenglättung |
| France | 8+ | NEBEF, Effacement, RTE Mechanism |
| Netherlands | 6+ | TenneT FCR, aFRR, Emergency Power |
| Other EU | 30+ | Various national programs |

**Revenue Projection Formula:**

```
Annual_Revenue = (
    Capacity_Payment * (committed_kW * $/kW-month * 12) +
    Energy_Payment * (avg_events_per_year * avg_curtailment_MWh * $/MWh) +
    Ancillary_Payment * (availability_hours * committed_kW * $/kW-hr) -
    Penalty_Risk * (P(shortfall) * penalty_per_event * expected_events)
)
```

**Key Models:**
- `DRProgram` - Program ID, name, ISO/RTO, program type, compensation, requirements, baseline methodology, penalties, enrollment window, event frequency
- `ProgramEligibility` - Facility meets minimum curtailment, metering requirements, telemetry, response time
- `RevenueProjection` - Expected capacity, energy, ancillary revenue, penalty risk, net annual revenue
- `ProgramPortfolio` - Selected programs with stacking analysis, exclusivity conflicts, combined revenue

### 4.3 Engine 3: Baseline Engine

**Purpose:** Simulate customer baseline load (CBL) using 8 industry-standard methodologies.

**Baseline Methodologies:**

| # | Methodology | Program | Description |
|---|-------------|---------|-------------|
| 1 | High 4 of 5 | PJM | Average of highest 4 of previous 5 non-event business days, +/- same-day symmetric additive adjustment |
| 2 | 10 of 10 | CAISO | Average of previous 10 non-event business days |
| 3 | High 5 Similar Days | NYISO | Average of 5 highest-consumption similar days in past 45 business days |
| 4 | 10 CP | ERCOT | Average of 10 highest coincident peak hours in 4 summer months (June-September) |
| 5 | Deemed Profile | UK | Standardized load profile based on customer class and meter point |
| 6 | Type I Regression | ISO-NE | Weather-regression baseline with day-type and occupancy adjustment |
| 7 | EU Standard Profile | EU | Reference load profile per customer segment with temperature correction |
| 8 | Custom Regression | Custom | Multivariate regression on temperature, occupancy, production, day-type |

**Baseline Optimization:**

```
Pre-event consumption strategy:
  For "High 4 of 5" baseline:
    - Increase consumption in qualifying hours on baseline-setting days
    - Shift non-critical loads INTO baseline period
    - Run pre-cooling/pre-heating during baseline hours
    - Constraint: must comply with program anti-gaming rules

  Baseline_Optimized = Baseline_Standard * (1 + optimization_uplift)
  Where optimization_uplift = 5-20% (program-dependent, within rules)
```

**Key Models:**
- `BaselineInput` - Historical interval data (15-min), weather data, event dates, program methodology selection
- `BaselineResult` - Calculated baseline kW per interval, adjustment factors, optimization opportunities, risk assessment
- `BaselineComparison` - Side-by-side comparison of all 8 methodologies with projected curtailment under each
- `BaselineRisk` - Probability distribution of baseline outcomes, under-performance risk, revenue sensitivity

### 4.4 Engine 4: Dispatch Optimizer Engine

**Purpose:** Multi-objective optimization for DR event load curtailment.

**Optimization Formulation:**

```
Minimize:
  w1 * Σ(criticality_i * curtailment_i) +        # Operational disruption
  w2 * Σ(comfort_deviation_i^2) +                  # Comfort impact (quadratic penalty)
  w3 * Σ(rebound_magnitude_i * curtailment_i) +   # Rebound severity
  w4 * (target_kW - Σ(curtailment_i))^2            # Target shortfall penalty

Subject to:
  Σ(curtailment_i) >= target_kW                    # Meet curtailment target
  curtailment_i <= max_curtailment_i   for all i   # Individual load limits
  curtailment_i >= 0                   for all i   # Non-negativity
  temperature_zone_j >= T_min_j        for all j   # Comfort lower bounds
  temperature_zone_j <= T_max_j        for all j   # Comfort upper bounds
  runtime_i >= min_runtime_i           for all i   # Equipment minimum runtime
  ramp_rate_i <= max_ramp_i            for all i   # Ramp rate limits
  critical_load_k = 0                  for all k   # Critical load protection
```

**Key Models:**
- `DispatchInput` - Curtailment target kW, event duration, notification time, load flexibility register, DER availability, weather forecast, comfort constraints
- `DispatchPlan` - Ordered curtailment sequence with timing, per-load kW reduction, DER dispatch, pre-conditioning commands
- `DispatchObjective` - Objective function weights, constraint violations, solution quality metrics
- `ReboundForecast` - Post-event load recovery profile, peak rebound kW, rebound duration, mitigation strategy

### 4.5 Engine 5: Event Manager Engine

**Purpose:** Real-time DR event lifecycle management.

**Event Lifecycle Phases:**

| Phase | Duration | Actions | Outputs |
|-------|----------|---------|---------|
| Notification | Immediate | Parse dispatch signal, validate event parameters, check program requirements | Event record, validity status |
| Preparation | 10 min - 24 hours | Run dispatch optimizer, activate pre-conditioning, verify DER readiness, notify stakeholders | Dispatch plan, readiness confirmation |
| Execution | 1 - 8 hours | Issue load control commands, monitor real-time performance, adjust dispatch dynamically | Load control log, performance metrics |
| Termination | 15 - 60 minutes | Release curtailed loads in staged sequence, manage rebound, return to normal | Restoration sequence, rebound profile |
| Assessment | Post-event | Calculate actual curtailment, compare to baseline, estimate revenue, identify improvements | Event performance report |

**Key Models:**
- `DREvent` - Event ID, program, dispatch time, start time, end time, target curtailment kW, duration, event type
- `EventExecution` - Phase status, curtailment achieved kW, performance ratio, DER contribution, alerts
- `LoadControlCommand` - Load ID, action (curtail/shed/shift/restore), magnitude, timestamp, confirmation status
- `EventAssessment` - Actual vs. baseline curtailment, revenue earned, penalties, compliance status

### 4.6 Engine 6: DER Coordinator Engine

**Purpose:** Coordinate distributed energy resources for DR events.

**DER Asset Types:**

| Asset Type | DR Contribution | Key Constraints |
|-----------|----------------|-----------------|
| Battery (BESS) | Discharge during event (sustained kW) | SOC limits (20-90%), cycling degradation, charge rate, round-trip efficiency |
| Solar PV | Offset load during daytime events | Weather-dependent, intermittent, only during daylight |
| Backup Generator | Supply load during events (if permitted) | Emissions permits, runtime limits (EPA 40 CFR 63), fuel supply, warm-up time |
| EV Chargers | Defer charging, V2B export | Smart charging required, customer priority, battery warranty impact |
| Thermal Storage | Pre-charge before event, discharge during | Capacity limits, discharge rate, efficiency losses |
| CHP/Cogen | Increase electrical output | Heat demand correlation, maintenance schedule, emissions |

**DER Dispatch Optimization:**

```
For each DER asset:
  availability_kw = f(SOC, capacity, constraints, time)
  contribution_kwh = availability_kw * event_duration * efficiency

Optimize DER dispatch to:
  Maximize: Σ(der_contribution_kw)
  Minimize: Σ(degradation_cost + fuel_cost + emissions_cost)
  Subject to: SOC constraints, cycling limits, emissions permits, fuel reserves
```

**Key Models:**
- `DERAsset` - Asset ID, type, rated capacity, current state, constraints, location
- `DERDispatch` - Asset ID, dispatch command, target output, duration, SOC trajectory
- `DERPerformance` - Asset ID, actual output, efficiency, degradation impact, fuel consumed
- `DERPortfolio` - Combined DER capacity, total contribution, coordination status

### 4.7 Engine 7: Performance Tracker Engine

**Purpose:** Track DR event performance against program requirements.

**Performance Metrics:**

| Metric | Calculation | Purpose |
|--------|------------|---------|
| Actual Curtailment (kW) | Baseline_kW - Actual_kW | Primary performance measure |
| Performance Ratio | Actual_Curtailment / Committed_Curtailment | Compliance assessment |
| Compliance Status | Performance_Ratio >= Program_Minimum | Pass/fail per event |
| Revenue Earned ($) | Capacity + Energy + Ancillary payments earned | Financial outcome |
| Penalties Incurred ($) | Non-performance penalty calculation per program rules | Financial risk |
| Season Compliance (%) | Events_Compliant / Total_Events * 100 | Seasonal program compliance |
| Curtailment Trend | Rolling average of performance ratio | Capacity degradation detection |

**Key Models:**
- `EventPerformance` - Event ID, baseline kW, actual kW, curtailment kW, performance ratio, compliance status, revenue, penalties
- `SeasonSummary` - Program, season, total events, compliant events, total curtailment MWh, total revenue, total penalties, net income
- `PerformanceTrend` - Rolling performance metrics, trend direction, capacity degradation alerts
- `ComplianceReport` - Program-specific compliance documentation for settlement

### 4.8 Engine 8: Revenue Optimizer Engine

**Purpose:** Optimize DR revenue across multiple programs and revenue streams.

**Revenue Streams:**

| Stream | Calculation | Typical Value |
|--------|-------------|---------------|
| Capacity Payment | committed_kW * rate_$/kW-month * months | $2-15/kW-month |
| Energy Payment | curtailment_MWh * rate_$/MWh * events | $100-500/MWh |
| Ancillary Service | availability_kW * hours * rate_$/kW-hr | $5-50/kW-year |
| Demand Charge Savings | peak_reduction_kW * demand_rate_$/kW | $5-25/kW-month |
| Price Arbitrage | shifted_kWh * price_differential | Variable |

**Revenue Optimization Formula:**

```
Net_Annual_Revenue = (
    Σ_programs(capacity_revenue + energy_revenue + ancillary_revenue) +
    demand_charge_savings +
    price_arbitrage_value -
    Σ_programs(expected_penalties) -
    enabling_technology_cost -
    operational_cost
)

ROI = Net_Annual_Revenue / (enabling_technology_investment + enrollment_cost)
```

**Key Models:**
- `RevenueStream` - Stream type, program, annual value, confidence level
- `RevenueForecast` - Annual gross revenue, penalties, net revenue, ROI, payback
- `WhatIfScenario` - Scenario name, parameter changes, projected revenue change
- `RevenueOptimization` - Optimal program portfolio, load allocation, DER dispatch strategy

### 4.9 Engine 9: Carbon Impact Engine

**Purpose:** Calculate carbon impact of DR using marginal emission factors.

**Marginal vs. Average Emission Factors:**

| Time Period | Average Grid EF (tCO2/MWh) | Marginal Grid EF (tCO2/MWh) | Ratio |
|------------|---------------------------|----------------------------|-------|
| Off-peak (night) | 0.25-0.35 | 0.30-0.45 | 1.2-1.3x |
| Mid-peak (day) | 0.30-0.40 | 0.40-0.55 | 1.3-1.4x |
| On-peak (afternoon) | 0.35-0.45 | 0.50-0.80 | 1.4-1.8x |
| Super-peak (heat wave) | 0.40-0.50 | 0.70-1.00 | 1.7-2.0x |

**Carbon Calculation:**

```
CO2e_avoided_per_event = curtailment_MWh * marginal_EF_tCO2_per_MWh

Annual_CO2e_avoided = Σ_events(CO2e_avoided_per_event)

Marginal_Abatement_Cost = Net_Revenue / Annual_CO2e_avoided
(Note: negative MAC means DR is profitable AND reduces emissions)
```

**Key Models:**
- `MarginalEmissionFactor` - Region, datetime, factor value (tCO2/MWh), source, confidence
- `EventCarbonImpact` - Event ID, curtailment MWh, marginal EF, CO2e avoided, scope breakdown
- `AnnualCarbonSummary` - Total CO2e avoided, by event, by program, SBTi contribution
- `CarbonReport` - GHG Protocol compliant report for Scope 2 emission reduction from DR

### 4.10 Engine 10: DR Reporting Engine

**Purpose:** Aggregate all DR analysis results into dashboards and reports.

**Dashboard Panels:**

| Panel | Content | Update Frequency |
|-------|---------|-----------------|
| Flexibility Profile | Load breakdown by curtailability, total capacity, seasonal variation | Monthly |
| Program Portfolio | Enrolled programs, commitments, event forecast | Monthly |
| Event History | Timeline of DR events with performance metrics | Per event |
| Revenue Tracker | Capacity + energy + ancillary payments, penalties, net income | Monthly |
| Carbon Impact | Marginal emission reductions per event, cumulative CO2e avoided | Per event |
| DER Performance | Battery SOC, solar contribution, EV flexibility utilization | Per event |
| Compliance Monitor | Program requirement tracking, shortfall alerts | Weekly |
| Revenue Forecast | 12-month forward projection based on program enrollment and event probability | Monthly |

**Key Models:**
- `DashboardData` - All panel data aggregated for rendering
- `ReportOutput` - Formatted report in requested format (MD, HTML, PDF, JSON, CSV)
- `ExecutiveSummary` - Top-line metrics, key events, revenue summary, carbon impact, recommendations
- `SettlementPackage` - Complete settlement documentation for program administrators

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Flexibility Assessment Workflow

**Purpose:** End-to-end load flexibility assessment from inventory through curtailment capacity.

**Phase 1: Load Inventory**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Import load inventory | Equipment list, BMS data, meter data | Categorized load register | <15 minutes |
| 1.2 | Classify load criticality | Load register, facility type preset | Criticality-tagged loads | <5 minutes |
| 1.3 | Import historical demand | Interval meter data (12+ months) | Demand profile with peaks | <10 minutes |

**Phase 2: Flexibility Scoring**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Assess curtailment capacity | Load profiles, equipment specifications | Per-load curtailment kW | <10 minutes |
| 2.2 | Evaluate response characteristics | Equipment controls, ramp rates, minimum runtime | Response time and duration limits | <5 minutes |
| 2.3 | Calculate flexibility scores | All assessments | Scored flexibility register | <5 minutes |

**Phase 3: Curtailment Capacity**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Aggregate by notification time | Flexibility register | Capacity matrix (time x duration) | <3 minutes |
| 3.2 | Apply seasonal adjustments | Capacity matrix, weather data | Seasonal capacity profiles | <3 minutes |
| 3.3 | Generate capacity curves | Adjusted capacity | Duration vs. curtailment curves | <2 minutes |

**Phase 4: Flexibility Report**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Generate flexibility profile | All assessments and capacity data | Comprehensive flexibility report | <5 minutes |
| 4.2 | Identify DR program readiness | Flexibility profile vs. program requirements | Program readiness assessment | <3 minutes |

**Acceptance Criteria:**
- [ ] All loads categorized by criticality level (0-4)
- [ ] Curtailment capacity calculated by notification time (5 tiers) and duration (5 tiers)
- [ ] Flexibility scores reproducible (SHA-256 verified)
- [ ] Total workflow duration <60 minutes for 500-load facility

### 5.2 Workflow 2: Program Enrollment Workflow

**Phases:** ProgramMatching -> RevenueProjection -> EnrollmentDocumentation -> CommitmentRegistration

**Acceptance Criteria:**
- [ ] All eligible programs identified with eligibility assessment
- [ ] Revenue projected for each program with penalty risk quantified
- [ ] Enrollment documentation pre-populated with facility data
- [ ] Stacking conflicts identified and resolved

### 5.3 Workflow 3: Event Preparation Workflow

**Phases:** EventNotification -> DispatchOptimization -> PreConditioningActivation

**Acceptance Criteria:**
- [ ] Dispatch signal parsed within 30 seconds of receipt
- [ ] Optimal curtailment plan generated within 2 minutes
- [ ] Pre-conditioning activated based on event start time and thermal model
- [ ] All stakeholders notified per escalation matrix

### 5.4 Workflow 4: Event Execution Workflow

**Phases:** LoadCurtailment -> RealTimeMonitoring -> PerformanceVerification -> LoadRestoration

**Acceptance Criteria:**
- [ ] Load control commands issued within 5 minutes of event start
- [ ] Real-time performance tracked at 1-minute intervals
- [ ] Performance shortfall alerts issued within 5 minutes of detection
- [ ] Staged load restoration prevents rebound peak exceeding 110% of baseline

### 5.5 Workflow 5: Settlement Workflow

**Phases:** BaselineCalculation -> PerformanceMeasurement -> RevenueSettlement

**Acceptance Criteria:**
- [ ] Baseline calculated per program-specific methodology
- [ ] Actual curtailment validated against meter data
- [ ] Revenue and penalties calculated with Decimal precision
- [ ] Settlement documentation generated in program-required format

### 5.6 Workflow 6: DER Optimization Workflow

**Phases:** AssetInventory -> DispatchStrategy -> CoordinatedResponse

**Acceptance Criteria:**
- [ ] All DER assets inventoried with current state and constraints
- [ ] Coordinated dispatch plan respects all asset constraints
- [ ] Combined DER contribution maximized within safety limits

### 5.7 Workflow 7: Reporting Workflow

**Phases:** DataAggregation -> ReportGeneration -> DistributionDelivery

**Acceptance Criteria:**
- [ ] Reports generated in all supported formats (MD, HTML, PDF, JSON, CSV)
- [ ] Executive summary <4 pages with key metrics
- [ ] Settlement reports match program administrator requirements
- [ ] SHA-256 provenance chain from meter data through final report

### 5.8 Workflow 8: Full DR Lifecycle Workflow

**Purpose:** Complete DR lifecycle from assessment through reporting.

**Phases:** FlexibilityAssessment -> ProgramSelection -> Enrollment -> EventPreparation -> EventExecution -> Settlement -> CarbonQuantification -> Reporting

**Acceptance Criteria:**
- [ ] All 8 phases execute with full data handoff
- [ ] Total lifecycle completion from new facility to first event settlement <4 weeks
- [ ] All outputs include SHA-256 provenance chain

---

## 6. Database Migrations

### 6.1 Migration Plan

| Migration ID | Description |
|-------------|-------------|
| `V286__pack037_demand_response_001` | Core DR schema: facility profiles, load inventory, flexibility assessments, curtailment capacity registers |
| `V287__pack037_demand_response_002` | DR program database: 200+ programs, compensation, requirements, baselines, penalties, enrollment |
| `V288__pack037_demand_response_003` | Baseline calculation tables: 8 methodologies, interval data, adjustment factors, simulation results |
| `V289__pack037_demand_response_004` | Dispatch optimization tables: curtailment plans, load control sequences, constraint parameters |
| `V290__pack037_demand_response_005` | Event management tables: events, phases, commands, performance metrics, event logs |
| `V291__pack037_demand_response_006` | DER coordination tables: asset registry, dispatch plans, SOC tracking, performance |
| `V292__pack037_demand_response_007` | Performance tracking tables: event performance, compliance, trends, alerts |
| `V293__pack037_demand_response_008` | Revenue tables: streams, forecasts, settlements, penalties, ROI analysis |
| `V294__pack037_demand_response_009` | Carbon impact tables: marginal emission factors, event carbon, annual summaries |
| `V295__pack037_demand_response_010` | Views, indexes, RLS policies, seed data, program database initialization |

---

## 7. Testing Strategy

### 7.1 Test Coverage

| Category | Tests | Coverage Target |
|----------|-------|-----------------|
| Unit tests (engines) | 400+ | 85% line coverage per engine |
| Integration tests | 150+ | All engine combinations and data flows |
| Workflow tests | 80+ | All 8 workflows end-to-end |
| Baseline methodology tests | 80+ | All 8 baselines with edge cases |
| Financial precision tests | 50+ | Decimal arithmetic verification |
| DER coordination tests | 60+ | Multi-asset dispatch scenarios |
| Performance tests | 30+ | Latency and throughput benchmarks |
| Total | 850+ | 85%+ overall coverage |

### 7.2 Test Fixtures

- Sample interval meter data (15-min, 12 months, multiple facilities)
- DR program database (representative subset of 200+ programs)
- Load flexibility profiles (per facility type preset)
- Historical DR events (various programs, performance outcomes)
- DER asset specifications (battery, solar, EV, thermal)
- Weather data (temperature, HDD/CDD for baseline normalization)
- Grid carbon intensity data (marginal emission factors by region/time)

---

## 8. Non-Functional Requirements

### 8.1 Performance

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Load flexibility assessment (500 loads) | <15 minutes | End-to-end assessment duration |
| Program matching (200+ programs) | <30 seconds | Time to evaluate all programs |
| Baseline calculation (12 months data) | <60 seconds per methodology | Per-methodology calculation time |
| Dispatch optimization (100 loads) | <30 seconds | Optimization solve time |
| Event response (notification to action) | <5 minutes | Critical path latency |
| Revenue calculation (annual) | <10 seconds | Full revenue reconciliation |
| Report generation | <60 seconds | Per-report generation time |
| Cache hit ratio | >80% | Program data, emission factors |
| Memory ceiling | 4096 MB | Maximum memory consumption |

### 8.2 Security

| Requirement | Implementation |
|-------------|---------------|
| Authentication | JWT (RS256) |
| Authorization | RBAC with facility-level and program-level access control |
| Encryption at rest | AES-256-GCM |
| Encryption in transit | TLS 1.3 |
| Audit logging | All DR events, dispatch commands, settlements logged |
| PII redaction | Facility addresses, utility account numbers redacted in reports |
| Data classification | INTERNAL, CONFIDENTIAL, RESTRICTED (grid signal data) |

### 8.3 Availability

| Requirement | Target |
|-------------|--------|
| System availability | 99.9% overall |
| Event window availability | 99.95% during DR event windows |
| Grid signal reception | <30 second latency |
| Failover | Active-passive with <60 second switchover |

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| CBL | Customer Baseline Load - estimated counterfactual consumption |
| DR | Demand Response |
| DER | Distributed Energy Resource |
| BESS | Battery Energy Storage System |
| V2B | Vehicle-to-Building |
| ISO/RTO | Independent System Operator / Regional Transmission Organization |
| LMP | Locational Marginal Price |
| OpenADR | Open Automated Demand Response protocol |
| CPP | Critical Peak Pricing |
| RTP | Real-Time Pricing |
| TOU | Time-of-Use |
| VPP | Virtual Power Plant |
| SOC | State of Charge (battery) |
| PUE | Power Usage Effectiveness (data centers) |
| ASHRAE A1 | ASHRAE thermal guideline class A1 (allowable IT environment) |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-23 | GreenLang Product Team | Initial PRD release |
