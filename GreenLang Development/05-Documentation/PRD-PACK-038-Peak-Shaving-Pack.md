# PRD-PACK-038: Peak Shaving Pack

**Pack ID:** PACK-038-peak-shaving
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-23
**Prerequisite:** None (standalone; enhanced with PACK-036 Utility Analysis Pack and PACK-037 Demand Response Pack if present; complemented by PACK-031/032/033 Energy Efficiency Packs)

---

## 1. Executive Summary

### 1.1 Problem Statement

Demand charges constitute 30-70% of commercial and industrial electricity bills in most utility territories worldwide. These charges are based on the highest 15-minute average power demand (kW) recorded during a billing period and can persist through "ratchet" mechanisms for 11-23 months. Despite this enormous financial impact, most organizations lack the analytical tools to understand, predict, and reduce their peak demand. Key challenges include:

1. **Opaque demand charge structures**: Utility tariffs contain complex tiered demand charges, time-of-use (TOU) demand periods, coincident peak (CP) charges, ratchet clauses, and power factor penalties. A typical commercial customer faces 3-5 overlapping demand charge components that interact in non-obvious ways. Without systematic tariff decomposition, facilities cannot identify which peaks drive which charges or quantify the value of peak reduction at different times.

2. **Unpredictable peak events**: Peak demand events are driven by weather (cooling/heating degree days), occupancy patterns, production schedules, equipment startup sequences, and random coincidences of multiple loads. Most facilities cannot predict when their next billing peak will occur, making proactive intervention impossible. Historical analysis shows that 60-80% of annual demand charges are set by just 5-10 peak events, yet these events are rarely anticipated.

3. **BESS sizing and economics uncertainty**: Battery Energy Storage Systems (BESS) are the primary technology for peak shaving, but optimal sizing requires analysis of hundreds of load profiles against dozens of tariff structures. Undersized batteries fail to capture peak events; oversized batteries have poor economics. Without dispatch simulation across 8,760+ hours with degradation modeling, organizations cannot determine the optimal battery capacity, power rating, or dispatch strategy for their specific load and tariff combination.

4. **Load shifting complexity**: Beyond batteries, peak shaving through load shifting (pre-cooling, thermal storage, production scheduling, EV charging deferral) requires coordination across multiple building systems. Each load has constraints (comfort bands, production deadlines, equipment limitations) that must be respected while minimizing the aggregate peak. Without multi-constraint optimization, load shifting attempts often create new peaks or violate operational requirements.

5. **Coincident peak (CP) management**: Many utilities and ISOs assess transmission charges based on a facility's demand during system-wide coincident peaks (e.g., PJM 5CP, ERCOT 4CP, ISO-NE ICL). These CP events occur unpredictably during extreme weather, and a single missed CP event can add $50,000-500,000 in annual transmission charges. Without CP prediction algorithms and automated response protocols, facilities are exposed to enormous financial risk from events that occur only 4-12 times per year.

6. **Ratchet demand traps**: Many tariffs include ratchet clauses where demand charges are based on 75-100% of the highest peak in the preceding 11-12 months. A single spike from equipment malfunction, startup sequencing error, or weather event can elevate demand charges for an entire year. Without ratchet analysis, impact quantification, and prevention strategies, organizations pay millions in excess charges from historical peaks they cannot undo.

7. **Power factor penalties**: Reactive power demand (kVAR) and poor power factor (<0.90) trigger penalty charges, increase apparent power demand (kVA), and reduce the effective capacity of electrical infrastructure. Power factor correction through capacitor banks, active harmonic filters, and VSD tuning can reduce demand charges by 5-15% and defer electrical infrastructure upgrades. Yet most facilities do not monitor or optimize power factor.

8. **Missing financial justification**: Peak shaving investments (BESS, load controls, power factor correction) require rigorous financial analysis including demand charge savings projections, battery degradation economics, incentive capture (ITC, SGIP, state programs), demand response revenue stacking, and risk-adjusted ROI. Without these analyses, capital approval committees cannot evaluate peak shaving proposals against competing investments.

### 1.2 Solution Overview

PACK-038 is the **Peak Shaving Pack** -- the eighth pack in the "Energy Efficiency Packs" category. While PACK-037 (Demand Response) focuses on grid-side DR program participation, PACK-038 focuses specifically on behind-the-meter peak demand reduction to minimize demand charges and optimize electrical infrastructure utilization.

The pack provides automated load profile analysis with peak identification across 12-36 months of interval data, comprehensive demand charge decomposition across tiered/TOU/CP/ratchet structures, BESS sizing optimization with degradation modeling and dispatch simulation, multi-load shifting optimization with constraint satisfaction, coincident peak prediction and automated response planning, ratchet demand analysis with prevention strategies, power factor analysis and correction sizing, and full financial modeling with incentive capture.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete peak shaving lifecycle from analysis through implementation and verification.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-038 Peak Shaving Pack |
|-----------|------------------------------|----------------------------|
| Load profile analysis | Manual spreadsheet review | Automated 15-min interval analysis across 12-36 months |
| Peak identification | Visual inspection | Statistical peak detection with clustering and attribution |
| Demand charge decomposition | Simple bill review | Full tariff modeling (tiered/TOU/CP/ratchet/PF penalties) |
| BESS sizing | Vendor proposal review | 8,760-hour dispatch simulation with degradation modeling |
| Load shifting analysis | Ad hoc adjustments | Multi-constraint optimization with comfort/production bounds |
| CP management | Reactive response | Predictive algorithms with automated response protocols |
| Ratchet analysis | Not performed | 12-month rolling analysis with prevention value quantification |
| Power factor | Utility penalty review | Harmonic analysis, capacitor bank sizing, VSD tuning |
| Financial analysis | Simple payback only | NPV, IRR, LCOE, incentive capture, revenue stacking |
| Audit trail | Spreadsheet-based | SHA-256 provenance, full calculation lineage |

### 1.4 Target Users

**Primary:**
- Facilities and energy managers seeking to reduce demand charges at commercial and industrial facilities
- Building owners and property managers optimizing electricity costs across portfolios
- Energy consultants and ESCOs designing peak shaving solutions for clients
- Battery storage developers evaluating behind-the-meter BESS economics

**Secondary:**
- Utility account managers analyzing customer demand profiles
- Sustainability managers quantifying peak shaving carbon benefits
- Finance teams evaluating BESS and peak shaving capital investments
- Grid operators assessing behind-the-meter peak reduction potential

### 1.5 Regulatory and Standards Framework

| Standard/Regulation | Relevance to Peak Shaving |
|---------------------|--------------------------|
| FERC Order 2222 | DER aggregation and wholesale market participation for BESS |
| IRS Section 48 (ITC) | Investment Tax Credit for battery storage (30-50% with adders) |
| SGIP (California) | Self-Generation Incentive Program for behind-the-meter storage |
| IEEE 1547-2018 | Standard for interconnection of DER including BESS |
| NFPA 855 | Standard for installation of stationary energy storage systems |
| UL 9540/9540A | Safety standards for energy storage systems |
| IEC 62933 | Electrical energy storage systems standards |
| ASHRAE 90.1 | Energy Standard for Buildings (demand management provisions) |
| ISO 50001:2018 | Energy management system (peak demand as significant energy use) |
| EU Energy Efficiency Directive (EED) 2023/1791 | Peak demand reduction as energy efficiency measure |
| GHG Protocol Scope 2 | Carbon impact of peak shaving (marginal vs average grid factors) |
| CSRD/ESRS E1 | Climate disclosure of energy management measures |

---

## 2. Technical Architecture

### 2.1 Pack Structure

```
packs/energy-efficiency/PACK-038-peak-shaving/
├── __init__.py
├── pack.yaml
├── engines/
│   ├── __init__.py
│   ├── load_profile_engine.py          # Engine 1: Load profile analysis
│   ├── peak_identifier_engine.py       # Engine 2: Peak detection & attribution
│   ├── demand_charge_engine.py         # Engine 3: Tariff decomposition
│   ├── bess_sizing_engine.py           # Engine 4: Battery optimization
│   ├── load_shifting_engine.py         # Engine 5: Load shift optimization
│   ├── cp_management_engine.py         # Engine 6: Coincident peak management
│   ├── ratchet_analysis_engine.py      # Engine 7: Ratchet demand analysis
│   ├── power_factor_engine.py          # Engine 8: Power factor correction
│   ├── financial_engine.py             # Engine 9: Financial modeling
│   └── peak_reporting_engine.py        # Engine 10: Dashboards & reports
├── workflows/
│   ├── __init__.py
│   ├── load_analysis_workflow.py       # Workflow 1: Load profile analysis
│   ├── peak_assessment_workflow.py     # Workflow 2: Peak assessment
│   ├── bess_optimization_workflow.py   # Workflow 3: BESS optimization
│   ├── load_shift_workflow.py          # Workflow 4: Load shifting
│   ├── cp_response_workflow.py         # Workflow 5: CP response
│   ├── implementation_workflow.py      # Workflow 6: Implementation planning
│   ├── verification_workflow.py        # Workflow 7: M&V verification
│   └── full_peak_shaving_workflow.py   # Workflow 8: Full lifecycle
├── templates/
│   ├── __init__.py
│   ├── load_profile_report.py          # Template 1: Load profile analysis
│   ├── peak_analysis_report.py         # Template 2: Peak identification
│   ├── demand_charge_report.py         # Template 3: Demand charge breakdown
│   ├── bess_sizing_report.py           # Template 4: BESS optimization
│   ├── load_shifting_report.py         # Template 5: Load shifting plan
│   ├── cp_management_report.py         # Template 6: CP management
│   ├── financial_analysis_report.py    # Template 7: Financial analysis
│   ├── power_factor_report.py          # Template 8: Power factor analysis
│   ├── executive_summary_report.py     # Template 9: Executive summary
│   └── verification_report.py          # Template 10: M&V verification
├── integrations/
│   ├── __init__.py
│   ├── pack_orchestrator.py            # Integration 1: Pipeline orchestrator
│   ├── mrv_bridge.py                   # Integration 2: MRV emissions bridge
│   ├── data_bridge.py                  # Integration 3: DATA agent bridge
│   ├── meter_data_bridge.py            # Integration 4: AMI/interval data
│   ├── utility_rate_bridge.py          # Integration 5: Tariff data integration
│   ├── bess_control_bridge.py          # Integration 6: Battery management
│   ├── bms_bridge.py                   # Integration 7: BMS load control
│   ├── pack036_bridge.py              # Integration 8: PACK-036 Utility Analysis
│   ├── pack037_bridge.py              # Integration 9: PACK-037 Demand Response
│   ├── health_check.py                 # Integration 10: System health
│   ├── setup_wizard.py                 # Integration 11: Configuration wizard
│   └── alert_bridge.py                 # Integration 12: Alert notifications
├── config/
│   ├── __init__.py
│   └── presets/
│       ├── __init__.py
│       ├── commercial_office.yaml
│       ├── manufacturing.yaml
│       ├── retail_grocery.yaml
│       ├── warehouse_cold.yaml
│       ├── healthcare.yaml
│       ├── data_center.yaml
│       ├── university_campus.yaml
│       └── mixed_use_portfolio.yaml
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_load_profile_engine.py
    ├── test_peak_identifier_engine.py
    ├── test_demand_charge_engine.py
    ├── test_bess_sizing_engine.py
    ├── test_load_shifting_engine.py
    ├── test_cp_management_engine.py
    ├── test_ratchet_analysis_engine.py
    ├── test_power_factor_engine.py
    ├── test_financial_engine.py
    ├── test_peak_reporting_engine.py
    ├── test_workflows.py
    └── test_integrations.py
```

### 2.2 Agent Dependencies

PACK-038 composes existing platform agents (zero duplication):

| Agent Layer | Agents Used | Purpose |
|-------------|-------------|---------|
| MRV | MRV-009 (Location-Based), MRV-010 (Market-Based), MRV-013 (Dual Reporting) | Scope 2 carbon from peak shaving |
| DATA | DATA-001 (PDF), DATA-002 (Excel/CSV), DATA-003 (ERP), DATA-010 (Quality Profiler), DATA-014 (Gap Filler), DATA-015 (Reconciliation) | Interval data intake and quality |
| FOUND | FOUND-001 (Orchestrator), FOUND-002 (Schema), FOUND-003 (Units), FOUND-004 (Assumptions), FOUND-005 (Citations), FOUND-008 (Reproducibility) | Pipeline foundation |
| Packs | PACK-036 (Utility Analysis), PACK-037 (Demand Response) | Rate structures, DR revenue stacking |

---

## 3. Engine Specifications

### 3.1 Engine 1: Load Profile Engine (`load_profile_engine.py`)

**Purpose:** Analyze 15-minute interval load data (8,760-35,040 intervals per year) to characterize demand patterns, identify load shape clusters, calculate statistical metrics, and detect anomalies.

**Key Calculations:**
- Load duration curve construction (sorted demand descending, 8,760 points)
- Load factor = average demand / peak demand (typical: 0.40-0.75)
- Diversity factor = sum of individual peaks / system peak
- Coincidence factor = system peak / sum of individual peaks
- Peak-to-average ratio = peak demand / average demand
- Standard deviation, coefficient of variation, percentile analysis (P50/P90/P95/P99)
- Day-type clustering (weekday/weekend/holiday patterns)
- Seasonal decomposition (summer/winter/shoulder profiles)
- Time-of-use period analysis (on-peak/mid-peak/off-peak)
- Anomaly detection (Z-score > 3.0 or IQR method)

**Reference Data (Zero-Hallucination):**
- Load factor benchmarks by facility type (ASHRAE/DOE CBECS)
- Day-type classification rules by ISO calendar
- TOU period definitions by utility territory

### 3.2 Engine 2: Peak Identifier Engine (`peak_identifier_engine.py`)

**Purpose:** Detect, classify, and attribute peak demand events across billing periods using statistical methods.

**Key Calculations:**
- Billing peak identification (highest 15-min demand per billing period)
- Top-N peak ranking with coincidence analysis
- Peak attribution by load category (HVAC, lighting, process, etc.)
- Peak clustering (temporal proximity analysis)
- Peak recurrence probability (Poisson/historical frequency)
- Weather-correlated peak analysis (CDD/HDD regression)
- Equipment startup peak detection (ramp rate analysis)
- Stochastic peak simulation (Monte Carlo, 1,000+ scenarios)
- Peak avoidability classification (avoidable/partially/unavoidable)
- Shaving potential per peak event (kW reduction achievable)

### 3.3 Engine 3: Demand Charge Engine (`demand_charge_engine.py`)

**Purpose:** Decompose utility bills into demand charge components with full tariff modeling.

**Key Calculations:**
- Flat demand charge calculation ($/kW x billing peak)
- Tiered/block demand charges (multiple rate tiers)
- TOU demand charges (on-peak, mid-peak, off-peak separate peaks)
- Seasonal demand charges (summer/winter rate differentials)
- Coincident peak (CP) charges (transmission/capacity based on CP events)
- Ratchet demand calculation (max of current vs % of prior 11-month peak)
- Power factor penalty calculation (kVA vs kW billing)
- Reactive power charges ($/kVAR for excess reactive demand)
- Demand charge attribution by component
- Marginal value of peak reduction ($/kW saved at each peak level)

**Reference Data (Zero-Hallucination):**
- Tariff structures for major US/EU utility territories
- CP event calendars by ISO/RTO region
- Ratchet clause parameters by tariff

### 3.4 Engine 4: BESS Sizing Engine (`bess_sizing_engine.py`)

**Purpose:** Optimize battery energy storage system sizing for peak shaving using 8,760-hour dispatch simulation.

**Key Calculations:**
- Optimal capacity (kWh) and power rating (kW) determination
- 8,760-hour dispatch simulation with 15-min resolution
- Peak shaving dispatch algorithm (target peak threshold)
- Depth of discharge (DoD) optimization (20-100% range)
- Calendar and cycle degradation modeling (Ah throughput method)
- Round-trip efficiency impact (85-95% range by chemistry)
- C-rate constraints and thermal derating
- Multi-year simulation (10-20 year project life)
- Sensitivity analysis on battery sizing parameters
- Technology comparison (Li-ion NMC, LFP, flow batteries)

**Reference Data (Zero-Hallucination):**
- Battery chemistry specifications (NMC, LFP, NCA, flow)
- Degradation curves by chemistry and operating conditions
- Cost benchmarks ($/kWh installed by chemistry and scale)
- Efficiency curves by C-rate and temperature

### 3.5 Engine 5: Load Shifting Engine (`load_shifting_engine.py`)

**Purpose:** Optimize load shifting strategies to reduce peak demand while respecting operational constraints.

**Key Calculations:**
- Shiftable load identification and quantification
- Pre-cooling/pre-heating optimization (thermal mass utilization)
- Production scheduling optimization (batch process timing)
- EV charging schedule optimization (fleet deferral)
- Thermal storage dispatch (ice storage, chilled water)
- Multi-load coordination with aggregate peak constraint
- Comfort boundary enforcement (ASHRAE 55 compliance)
- Constraint satisfaction (production deadlines, equipment limits)
- Rebound effect estimation after load restoration
- Combined BESS + load shifting optimization

### 3.6 Engine 6: CP Management Engine (`cp_management_engine.py`)

**Purpose:** Predict and manage coincident peak events to minimize transmission and capacity charges.

**Key Calculations:**
- CP event prediction using weather, load, and grid data
- Historical CP event analysis (timing, magnitude, frequency)
- Probability scoring for upcoming CP windows
- Alert threshold calibration (false positive vs missed event tradeoff)
- Auto-curtailment target calculation during CP events
- CP tag value calculation ($/kW of CP demand)
- ICAP tag calculation for ISO capacity markets
- Multi-zone CP management for portfolio facilities
- CP event response performance tracking
- Annual CP charge forecast with confidence intervals

**Reference Data (Zero-Hallucination):**
- PJM 5CP methodology and historical CP dates
- ERCOT 4CP methodology and historical CP dates
- ISO-NE ICL (Installed Capacity Load) methodology
- NYISO ICAP tag methodology
- CP tag values by ISO/RTO ($2-$20/kW-month range)

### 3.7 Engine 7: Ratchet Analysis Engine (`ratchet_analysis_engine.py`)

**Purpose:** Analyze ratchet demand clauses and quantify the financial impact of peak demand spikes.

**Key Calculations:**
- Ratchet demand identification from 12-month rolling history
- Ratchet reset date and remaining ratchet period calculation
- Financial impact of ratchet demand (excess charges per month)
- Ratchet prevention value (savings from avoiding new peak)
- Spike root cause analysis (weather, equipment, startup)
- Equipment startup sequence optimization to avoid ratchet triggers
- Ratchet decay projection (months until ratchet release)
- Multi-ratchet interaction analysis (summer/winter ratchets)
- Break-even analysis for ratchet reduction investments
- Alarm threshold calculation for ratchet prevention

### 3.8 Engine 8: Power Factor Engine (`power_factor_engine.py`)

**Purpose:** Analyze power factor and reactive power demand to size correction equipment.

**Key Calculations:**
- Power factor calculation from kW, kVAR, kVA measurements
- Power factor profile across 24-hour and seasonal cycles
- Reactive power demand analysis by load category
- Capacitor bank sizing (fixed and automatic switching)
- Active harmonic filter sizing for non-linear loads
- kVA vs kW billing impact analysis
- Power factor penalty quantification and avoidance savings
- Harmonic distortion analysis (THD percentage)
- Resonance risk assessment for capacitor installation
- Infrastructure capacity recovery from PF improvement

**Reference Data (Zero-Hallucination):**
- Power factor penalty schedules by utility territory
- Capacitor bank specifications and costs
- Harmonic filter specifications
- Power factor benchmarks by facility type

### 3.9 Engine 9: Financial Engine (`financial_engine.py`)

**Purpose:** Comprehensive financial modeling for peak shaving investments.

**Key Calculations:**
- Demand charge savings projection (monthly/annual)
- BESS investment analysis (NPV, IRR, payback, LCOE)
- ITC and incentive capture modeling (IRS Section 48, SGIP, state programs)
- Battery degradation financial impact (capacity fade economics)
- Revenue stacking (peak shaving + DR + ancillary services + arbitrage)
- Sensitivity analysis on key financial parameters
- Monte Carlo risk analysis (demand uncertainty, rate escalation)
- Levelized Cost of Storage (LCOS) calculation
- Comparison analysis (BESS vs load shifting vs PF correction)
- Total cost of ownership (TCO) over project lifetime

**Reference Data (Zero-Hallucination):**
- ITC rates and adder qualifications (Section 48/48E)
- SGIP incentive rates by step and category
- State-level storage incentive programs (20+ states)
- Demand charge escalation rates by region
- Battery replacement cost projections

### 3.10 Engine 10: Peak Reporting Engine (`peak_reporting_engine.py`)

**Purpose:** Generate dashboards, reports, and executive summaries for peak shaving analysis.

**Key Calculations:**
- 8 dashboard panels (load profile, peak events, demand charges, BESS dispatch, load shifting, CP status, power factor, financial summary)
- 7 report types (load analysis, peak assessment, BESS sizing, load shifting, CP management, financial analysis, verification)
- KPI calculations (peak reduction %, demand charge savings, BESS utilization, ROI)
- Trend analysis (month-over-month, year-over-year peak comparison)
- Multi-format export (Markdown, HTML, JSON, CSV)
- Provenance hashing for all report outputs

---

## 4. Workflow Specifications

### 4.1 Workflow 1: Load Analysis Workflow
- **Phase 1: Data Intake** - Import 15-min interval data via AMI, Green Button, or CSV
- **Phase 2: Profile Analysis** - Calculate load statistics, duration curve, day-type clusters
- **Phase 3: Peak Identification** - Detect and classify billing peaks
- **Phase 4: Baseline Establishment** - Set demand baseline for savings measurement

### 4.2 Workflow 2: Peak Assessment Workflow
- **Phase 1: Peak Attribution** - Attribute peaks to load categories
- **Phase 2: Demand Charge Decomposition** - Full tariff analysis
- **Phase 3: Avoidability Assessment** - Classify peak reduction potential
- **Phase 4: Strategy Recommendation** - Recommend peak shaving approaches

### 4.3 Workflow 3: BESS Optimization Workflow
- **Phase 1: Load Characterization** - Size requirements analysis
- **Phase 2: Technology Selection** - Chemistry and configuration comparison
- **Phase 3: Dispatch Simulation** - 8,760-hour simulation with degradation
- **Phase 4: Financial Analysis** - NPV, IRR, incentives, revenue stacking

### 4.4 Workflow 4: Load Shift Workflow
- **Phase 1: Shiftable Load Inventory** - Identify and quantify loads
- **Phase 2: Constraint Mapping** - Define operational boundaries
- **Phase 3: Schedule Optimization** - Multi-load coordination

### 4.5 Workflow 5: CP Response Workflow
- **Phase 1: CP Prediction** - Weather and grid signal analysis
- **Phase 2: Response Planning** - Curtailment target calculation
- **Phase 3: Event Execution** - Automated response coordination

### 4.6 Workflow 6: Implementation Workflow
- **Phase 1: Solution Design** - Detailed engineering specification
- **Phase 2: Procurement** - Vendor comparison and selection
- **Phase 3: Commissioning** - Installation verification

### 4.7 Workflow 7: Verification Workflow
- **Phase 1: Baseline Comparison** - Pre/post demand comparison
- **Phase 2: Savings Calculation** - Verified demand charge reduction
- **Phase 3: Performance Reporting** - M&V documentation

### 4.8 Workflow 8: Full Peak Shaving Lifecycle Workflow
- **8-phase master workflow** orchestrating all 7 sub-workflows in sequence

---

## 5. Database Migrations (V296-V305)

| Migration | Description | Key Tables |
|-----------|-------------|------------|
| V296 | Load profiles and interval data | `ps_load_profiles`, `ps_interval_data`, `ps_load_statistics`, `ps_day_type_clusters` |
| V297 | Peak identification and attribution | `ps_peak_events`, `ps_peak_attribution`, `ps_peak_clusters`, `ps_peak_simulation` |
| V298 | Demand charge decomposition | `ps_tariff_structures`, `ps_demand_charges`, `ps_charge_components`, `ps_marginal_values` |
| V299 | BESS sizing and dispatch | `ps_bess_configurations`, `ps_dispatch_simulations`, `ps_degradation_tracking`, `ps_technology_comparisons` |
| V300 | Load shifting optimization | `ps_shiftable_loads`, `ps_shift_schedules`, `ps_constraint_definitions`, `ps_coordination_plans` |
| V301 | Coincident peak management | `ps_cp_events`, `ps_cp_predictions`, `ps_cp_responses`, `ps_cp_charges` |
| V302 | Ratchet demand analysis | `ps_ratchet_history`, `ps_ratchet_impacts`, `ps_spike_analysis`, `ps_prevention_plans` |
| V303 | Power factor analysis | `ps_power_factor_data`, `ps_reactive_demand`, `ps_correction_sizing`, `ps_pf_penalties` |
| V304 | Financial analysis | `ps_financial_models`, `ps_incentive_capture`, `ps_revenue_stacking`, `ps_sensitivity_results` |
| V305 | Views, indexes, audit trail, seed data | Materialized views, audit trail, seeded tariff data, CP calendars |

---

## 6. Testing Strategy

### 6.1 Test Coverage Targets
- **Unit tests:** 850+ test functions across 12 test files
- **Parametrize expansions:** 400+ for multi-scenario coverage
- **Total expanded tests:** 1,250+ test cases
- **Coverage target:** 85%+ line coverage

### 6.2 Key Test Scenarios

| Engine | Key Test Scenarios |
|--------|--------------------|
| Load Profile | 15-min/30-min/60-min intervals, missing data handling, leap years, DST transitions |
| Peak Identifier | Single peaks, clustered peaks, weather-correlated, equipment startups, seasonal |
| Demand Charge | Flat/tiered/TOU/CP/ratchet/PF charges, multi-component bills |
| BESS Sizing | NMC/LFP/flow, 100kWh-10MWh range, degradation over 10-20 years |
| Load Shifting | HVAC/production/EV/thermal, constraint satisfaction, rebound |
| CP Management | PJM 5CP, ERCOT 4CP, ISO-NE ICL, multi-zone, prediction accuracy |
| Ratchet | 12-month rolling, summer/winter ratchets, spike prevention |
| Power Factor | 0.70-1.00 range, capacitor/filter sizing, resonance checks |
| Financial | NPV/IRR/payback, ITC 30-50%, SGIP steps, revenue stacking |
| Reporting | MD/HTML/JSON formats, provenance hashing, KPI accuracy |

---

## 7. Non-Functional Requirements

### 7.1 Performance
- Load profile analysis: <30 seconds for 35,040 intervals (1 year at 15-min)
- BESS dispatch simulation: <60 seconds for 8,760 hours
- CP prediction: <5 seconds per evaluation window
- Report generation: <10 seconds per report

### 7.2 Security
- Multi-tenant data isolation via PostgreSQL RLS
- SHA-256 provenance hashing on all calculation outputs
- Encryption at rest for interval data and financial models
- RBAC integration for pack operations

### 7.3 Accuracy
- Decimal arithmetic for all financial calculations
- 15-minute resolution for demand calculations
- ±2% accuracy on BESS degradation projections
- ±5% accuracy on demand charge calculations

---

## 8. Glossary

| Term | Definition |
|------|-----------|
| BESS | Battery Energy Storage System |
| CBL | Customer Baseline Load |
| CP | Coincident Peak - system-wide peak demand event |
| DoD | Depth of Discharge - percentage of battery capacity used |
| ICAP | Installed Capacity - ISO capacity market tag |
| ITC | Investment Tax Credit (US federal, IRS Section 48/48E) |
| kVA | Kilovolt-ampere - apparent power (includes reactive component) |
| kVAR | Kilovolt-ampere reactive - reactive power demand |
| kW | Kilowatt - real power demand |
| LCOS | Levelized Cost of Storage |
| LFP | Lithium Iron Phosphate battery chemistry |
| NMC | Nickel Manganese Cobalt battery chemistry |
| PF | Power Factor = kW / kVA (0 to 1.0) |
| Ratchet | Billing mechanism using historical peak for demand charges |
| SGIP | Self-Generation Incentive Program (California storage incentive) |
| SOC | State of Charge - current battery charge level |
| THD | Total Harmonic Distortion |
| TOU | Time-of-Use - rate period classification |
| 5CP | PJM Five Coincident Peak methodology |
| 4CP | ERCOT Four Coincident Peak methodology |
