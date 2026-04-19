# Product Requirements Document: GL-018 FLUEFLOW

**Version:** 1.0
**Date:** 2024-12-02
**Product Manager:** GreenLang Product Team
**Status:** Approved for Development

---

## 1. Executive Summary

### Problem Statement

Industrial combustion systems (boilers, furnaces, incinerators) worldwide waste 5-15% of their fuel due to suboptimal combustion conditions and emit pollutants (NOx, CO, SOx) that violate increasingly stringent environmental regulations. Plant operators lack real-time visibility into combustion efficiency and emissions performance, relying on quarterly manual stack tests that cost $5,000-$15,000 per test and only provide a snapshot in time.

Current challenges:
- **Fuel waste:** 5-15% of fuel energy lost due to excess air, incomplete combustion, and stack heat loss
- **Emissions violations:** NOx and CO violations resulting in fines of $25,000-$100,000 per day
- **Manual analysis:** Stack testing is expensive, infrequent, and provides no actionable real-time insights
- **Reactive tuning:** Combustion tuning happens only after efficiency degrades or emissions violations occur
- **Limited expertise:** Shortage of combustion engineers to interpret flue gas data and optimize burners

### Solution Overview

GL-018 FLUEFLOW is an AI-powered flue gas analysis agent that continuously monitors combustion performance, calculates thermal efficiency using EPA Method 19 and ASME PTC 4.1 standards, and provides real-time optimization recommendations to minimize fuel consumption and emissions. The agent integrates with existing flue gas analyzers and SCADA/DCS systems to deliver:

1. **Real-Time Combustion Efficiency:** Continuous calculation of thermal efficiency (HHV/LHV basis) with heat loss breakdown
2. **Emissions Compliance Monitoring:** NOx, CO, SOx, CO2 tracking with automatic regulatory limit checking
3. **Air-Fuel Ratio Optimization:** Determination of optimal O2 setpoint to balance efficiency and emissions
4. **Burner Performance Diagnostics:** Fouling detection, incomplete combustion alerts, and maintenance recommendations
5. **Economic Analysis:** Fuel cost savings quantification and ROI tracking for optimization actions

Zero-Hallucination Guarantee: All combustion calculations use deterministic engineering formulas from EPA, ASME, and ISO standards. No AI is used for numeric calculations, ensuring 100% accuracy and regulatory defensibility.

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Fuel Savings | 5% average | Monthly fuel consumption reduction |
| Efficiency Improvement | +3 percentage points | Thermal efficiency (HHV) increase |
| NOx Reduction | 20% average | NOx emissions (lb/MMBtu) reduction |
| CO Reduction | 60% average | CO emissions reduction |
| Compliance Violations | 90% reduction | Quarterly violation count vs. baseline |
| Customer Payback Period | <18 months | Time to recover implementation cost |
| Annual Savings per Unit | $150,000 | Fuel + emissions savings per combustion unit |
| Customer Satisfaction (NPS) | >50 | Net Promoter Score survey |
| Revenue (Year 1) | $60M ARR | Annual recurring revenue |

---

## 2. Market & Competitive Analysis

### Market Opportunity

**Total Addressable Market (TAM):** $4B
- 2.5 million industrial combustion systems globally
- Boilers: 1.8M units
- Process furnaces: 400K units
- Incinerators: 200K units
- Other (dryers, ovens, kilns): 100K units

**Serviceable Addressable Market (SAM):** $1.2B
- Large industrial combustion systems (>10 MW thermal)
- Systems with existing flue gas analyzers
- Facilities subject to emissions regulations (EPA, EU IED)

**Serviceable Obtainable Market (SOM):** $60M (Year 1)
- Initial focus: North America and EU power plants, refineries, chemical plants
- Target: 400 combustion units in Year 1

### Target Customers

**Primary Segments:**

1. **Power Generation Plants**
   - Company size: 500-10,000 employees
   - Combustion units: 2-20 boilers per plant
   - Annual fuel cost: $20M-$500M
   - Pain points: NOx regulations, carbon taxes, fuel price volatility

2. **Refineries and Petrochemical Plants**
   - Company size: 1,000-50,000 employees
   - Combustion units: 10-100 fired heaters and boilers
   - Annual fuel cost: $50M-$1B
   - Pain points: Process uptime, emissions permits, energy intensity

3. **Chemical Manufacturing**
   - Company size: 500-20,000 employees
   - Combustion units: 5-50 process furnaces and boilers
   - Annual fuel cost: $10M-$200M
   - Pain points: Product quality, emissions, operational efficiency

4. **Pulp and Paper Mills**
   - Company size: 500-5,000 employees
   - Combustion units: 3-15 recovery boilers and power boilers
   - Annual fuel cost: $15M-$100M
   - Pain points: Fuel costs (30% of operating cost), carbon footprint

**Secondary Segments:**
- Food and beverage processing
- Glass and ceramics manufacturing
- Cement kilns
- Metal smelting and heat treating
- Hospitals and universities (central plants)

### Competitive Landscape

| Solution | Strengths | Weaknesses | FLUEFLOW Advantage |
|----------|-----------|------------|-------------------|
| **Manual Stack Testing** | No capital cost | Infrequent ($10K/test), no real-time data, reactive | Continuous monitoring, <18 mo payback |
| **DCS Built-in Efficiency Calc** | Already integrated | Simplified calculations, no diagnostics, no AI insights | EPA/ASME-compliant calculations, AI recommendations |
| **Combustion Optimization Software** (e.g., Honeywell, Emerson) | Proven technology | $200K-$1M cost, proprietary lock-in, complex tuning | $50K-$150K, open integration, zero-hallucination AI |
| **CEMS (Continuous Emissions Monitoring)** | Regulatory compliance | Emissions only, no efficiency optimization | Efficiency + emissions + optimization in one |
| **Energy Management Platforms** (e.g., Schneider, Siemens) | Broad scope | Generic (not combustion-specific), no real-time tuning | Purpose-built for combustion, real-time optimization |
| **Combustion Consultants** | Deep expertise | $500/hr-$2,000/hr, not scalable, infrequent visits | 24/7 AI expert, <10% of consultant cost |

**Key Differentiators:**
1. **Zero-Hallucination Guarantee:** All combustion calculations use deterministic formulas (EPA Method 19, ASME PTC 4.1)
2. **Real-Time Optimization:** Continuous monitoring vs. quarterly/annual testing
3. **Regulatory Compliance Built-In:** EPA, ASME, ISO 50001 standards embedded in calculations
4. **Rapid ROI:** <18 month payback via 5% fuel savings
5. **Open Integration:** Works with any flue gas analyzer and SCADA/DCS (OPC UA, Modbus)
6. **AI-Powered Insights:** Narrative explanations and recommendations (LLM for text only, never for calculations)

---

## 3. Features & Requirements

### Must-Have Features (P0 - Launch Blockers)

---

#### Feature 1: Real-Time Flue Gas Composition Monitoring

**User Story:**
```
As a combustion engineer,
I want to continuously monitor O2, CO2, CO, NOx, and stack temperature in real-time,
So that I can detect combustion issues immediately and avoid emissions violations.
```

**Acceptance Criteria:**
- [ ] Integrates with ABB, Siemens, Emerson, Yokogawa flue gas analyzers via Modbus TCP / OPC UA
- [ ] Polls analyzer data every 5 seconds with <10 second latency
- [ ] Displays real-time trends for O2, CO2, CO, NOx, SOx, stack temperature, moisture
- [ ] Automatically switches to backup analyzer if primary fails (<30 second failover)
- [ ] Validates data quality and flags sensor faults (out-of-range, stuck values)
- [ ] Stores raw data with 5-second resolution for 90 days (compressed to S3 after 30 days)

**Non-Functional Requirements:**
- Performance: <5 second data refresh, <10 second end-to-end latency
- Reliability: 99.9% uptime, automatic failover to backup analyzer
- Security: TLS 1.3 encryption for analyzer communication
- Usability: Grafana dashboard with 30-second refresh

**Dependencies:**
- Modbus TCP / OPC UA client libraries (asyncua, pymodbus)
- Time-series database (TimescaleDB or InfluxDB)
- Flue gas analyzer connectivity (customer network)

**Estimated Effort:** 3 weeks (1 integration engineer, 1 backend engineer)

**Regulatory Justification:** ISO 12039 requires continuous measurement of CO, CO2, O2 for emissions monitoring.

---

#### Feature 2: Combustion Efficiency Calculation (EPA Method 19 / ASME PTC 4.1)

**User Story:**
```
As a plant manager,
I want the system to calculate thermal efficiency using EPA Method 19 and ASME PTC 4.1 standards,
So that I can demonstrate regulatory compliance and track efficiency trends over time.
```

**Acceptance Criteria:**
- [ ] Calculates thermal efficiency (HHV and LHV basis) every 1 minute
- [ ] Implements EPA Method 19 heat loss calculation:
  - Heat loss via dry flue gas (sensible heat)
  - Heat loss via moisture in fuel
  - Heat loss via moisture from H2 combustion
  - Heat loss via incomplete combustion (CO, THC)
  - Heat loss via radiation and convection (ASME default or measured)
- [ ] Calculates efficiency using ASME PTC 4.1 input-output method
- [ ] Provides heat loss breakdown (% of total input) with each component
- [ ] Compares current efficiency to design/baseline efficiency
- [ ] Achieves <0.1 percentage point accuracy vs. manual calculations (validated against 100 known test cases)
- [ ] Generates SHA-256 provenance hash for each calculation (audit trail)

**Calculation Methodology:**

Thermal Efficiency (HHV basis):
```
η = 100% - Σ(Heat Losses)

Heat Losses:
1. Dry Flue Gas Loss (%) = (m_fg × Cp_fg × (T_fg - T_air)) / Q_fuel × 100
2. Moisture in Fuel Loss (%) = (m_H2O_fuel × h_fg(T_fg)) / Q_fuel × 100
3. H2 Combustion Moisture Loss (%) = (9 × m_H2_fuel × h_fg(T_fg)) / Q_fuel × 100
4. Incomplete Combustion Loss (%) = (m_CO × LHV_CO) / Q_fuel × 100
5. Radiation & Convection Loss (%) = ASME default curve or measured

Where:
- m_fg = flue gas mass flow (kg/hr)
- Cp_fg = specific heat of flue gas (kJ/kg·K)
- T_fg = flue gas temperature (K)
- T_air = ambient air temperature (K)
- Q_fuel = fuel heat input (kJ/hr, HHV basis)
- h_fg = enthalpy of vaporization at stack temperature
- LHV_CO = lower heating value of CO (10.1 MJ/kg)

All properties calculated using Cantera/CoolProp thermodynamic libraries.
```

**Edge Cases:**
- Missing fuel composition → Use fuel type defaults (ASME/EPA tables)
- Missing stack flow → Estimate from fuel flow and stoichiometry
- High moisture (>20%) → Use wet-basis corrections per EPA Method 19

**Non-Functional Requirements:**
- Accuracy: <0.1 percentage point vs. manual EPA Method 19 calculation
- Auditability: Complete calculation provenance with SHA-256 hashes
- Reproducibility: Bit-perfect (same input → same output)
- Performance: <100 ms calculation time per unit

**Estimated Effort:** 4 weeks (1 combustion engineer, 1 calculation engine developer)

**Regulatory Justification:** EPA Method 19 and ASME PTC 4.1 are industry-standard methods for efficiency testing and regulatory reporting.

---

#### Feature 3: Emissions Compliance Monitoring (NOx, CO, SOx, CO2)

**User Story:**
```
As an environmental compliance manager,
I want the system to track NOx, CO, SOx, and CO2 emissions in real-time and alert me if permit limits are exceeded,
So that I can avoid regulatory violations and fines.
```

**Acceptance Criteria:**
- [ ] Calculates NOx emissions in ppm @ 3% O2, lb/MMBtu, and kg/hr
- [ ] Calculates CO emissions in ppm @ 3% O2, lb/MMBtu, and kg/hr
- [ ] Calculates SOx emissions (if measured) in ppm, lb/MMBtu, and kg/hr
- [ ] Calculates CO2 emissions in kg/hr and tonnes/year
- [ ] Corrects measured concentrations to reference O2 level (3% for boilers, 15% for gas turbines)
- [ ] Compares emissions to permit limits (user-configurable per unit)
- [ ] Generates alerts if emissions exceed 90% of limit (warning) or 100% of limit (critical)
- [ ] Stores all emissions data for 7 years (regulatory requirement)
- [ ] Exports quarterly and annual emissions reports in EPA/EU IED format
- [ ] Validates CEMS data (if present) and flags discrepancies

**Emissions Calculation Methodology:**

NOx Correction to 3% O2:
```
NOx_corrected = NOx_measured × (20.9 - O2_ref) / (20.9 - O2_measured)

Where:
- NOx_measured = measured NOx concentration (ppm dry)
- O2_ref = reference O2 level (3% for boilers, 15% for gas turbines)
- O2_measured = measured O2 concentration (% dry)

NOx mass emission rate (lb/MMBtu):
NOx_lb_MMBtu = (NOx_ppm × MW_NOx × K) / (HHV_fuel × (20.9 / (20.9 - O2_measured)))

Where:
- MW_NOx = molecular weight of NOx (46 for NO2)
- K = conversion factor (depends on fuel type, per EPA Method 19)
- HHV_fuel = higher heating value of fuel (Btu/lb or Btu/scf)

CO2 mass emission rate (kg/hr):
CO2_kg_hr = (Fuel_flow_kg_hr × C_content × (44/12)) × Combustion_efficiency

Where:
- C_content = carbon content in fuel (kg C / kg fuel)
- 44/12 = molecular weight ratio (CO2 / C)
```

**Edge Cases:**
- SOx not measured → Calculate from fuel sulfur content if known
- Missing O2 → Use default excess air assumption (alert user)
- Negative draft pressure → Validate with barometric pressure

**Non-Functional Requirements:**
- Accuracy: ±5% vs. CEMS (Continuous Emissions Monitoring System)
- Compliance: EPA Part 60, EU IED reporting formats
- Auditability: Immutable 7-year storage of all emissions data
- Performance: <50 ms emissions calculation per unit

**Estimated Effort:** 3 weeks (1 environmental engineer, 1 backend engineer)

**Regulatory Justification:** 40 CFR Part 60 (EPA NSPS) and EU IED 2010/75/EU require continuous or periodic emissions monitoring and reporting.

---

#### Feature 4: Air-Fuel Ratio Optimization

**User Story:**
```
As a boiler operator,
I want the system to recommend the optimal O2 setpoint to minimize fuel consumption while staying compliant with emissions limits,
So that I can reduce fuel costs without risking NOx or CO violations.
```

**Acceptance Criteria:**
- [ ] Calculates current excess air percentage and lambda (air ratio)
- [ ] Determines stoichiometric air requirement based on fuel composition
- [ ] Recommends optimal O2 setpoint based on:
  - Minimum efficiency loss (excess air heat loss)
  - NOx limit compliance (lower O2 → higher NOx)
  - CO limit compliance (too low O2 → incomplete combustion)
  - Load and firing rate (optimal O2 varies with load)
- [ ] Displays current O2 deviation from optimal (percentage points)
- [ ] Provides air-fuel status assessment (OPTIMAL, EXCESS_AIR, INSUFFICIENT_AIR, CRITICAL_LOW_O2)
- [ ] Generates actionable recommendations (e.g., "Reduce O2 setpoint by 0.5% to save 1.2% fuel")
- [ ] Updates recommendations every 5 minutes based on changing conditions
- [ ] Logs all recommendations with provenance hash

**Optimization Logic:**

```python
# Simplified optimization logic (actual implementation more complex)

# 1. Calculate stoichiometric air
A_stoich = calculate_stoichiometric_air(fuel_composition)

# 2. Calculate current excess air
EA_current = ((O2_measured / (21 - O2_measured)) × 100)

# 3. Determine optimal excess air range
EA_min = 5%  # Minimum to avoid CO formation
EA_max = 30%  # Maximum before efficiency loss is excessive

# 4. Check NOx constraint
if NOx_current > NOx_limit × 0.9:
    EA_optimal = EA_current + 2%  # Increase air to reduce NOx

# 5. Check CO constraint
elif CO_current > 100 ppm:
    EA_optimal = EA_current + 3%  # Increase air to reduce CO

# 6. Otherwise, minimize heat loss
else:
    EA_optimal = EA_min + safety_margin(load, fuel_type)

# 7. Convert to O2 setpoint
O2_optimal = (21 × EA_optimal) / (100 + EA_optimal)

# 8. Calculate benefit
fuel_savings_percent = (EA_current - EA_optimal) × efficiency_sensitivity
```

**Edge Cases:**
- Low load (<30% MCR) → Higher optimal O2 due to poor mixing
- Fuel switching → Recalculate stoichiometric air immediately
- Burner in manual mode → Advisory-only recommendations (no automatic control)

**Non-Functional Requirements:**
- Safety: Never recommend O2 <1.5% (incomplete combustion risk)
- Performance: <200 ms optimization calculation
- Usability: Clear explanation of recommendation ("Why this setpoint?")

**Estimated Effort:** 3 weeks (1 combustion engineer, 1 optimization algorithm developer)

**Business Justification:** 1% excess air reduction = ~0.6% fuel savings. For a 100 MW boiler burning natural gas at $4/MMBtu, this saves $150,000/year.

---

#### Feature 5: Burner Performance Diagnostics

**User Story:**
```
As a maintenance engineer,
I want the system to detect burner fouling, incomplete combustion, and other performance issues,
So that I can schedule maintenance proactively before efficiency degrades or failures occur.
```

**Acceptance Criteria:**
- [ ] Calculates combustion quality score (0-100) based on:
  - O2 stability (low variance = good)
  - CO level (low CO = complete combustion)
  - NOx level (moderate NOx = good mixing)
  - Efficiency vs. baseline
- [ ] Detects burner fouling by monitoring O2 trend at constant load (rising O2 = fouling)
- [ ] Detects incomplete combustion (CO >400 ppm or THC >50 ppm)
- [ ] Assesses flame quality based on flue gas composition signature
- [ ] Tracks burner age and performance degradation over time
- [ ] Generates maintenance recommendations:
  - "Burner cleaning recommended (fouling detected)"
  - "Air register adjustment needed (high O2 variance)"
  - "Fuel nozzle inspection required (high CO)"
- [ ] Provides air register and damper position optimization suggestions
- [ ] Creates predictive maintenance work orders in CMMS (SAP PM, Maximo)

**Diagnostic Logic:**

```python
# Combustion Quality Score Calculation

# 1. O2 Stability Score (0-25 points)
o2_variance = std_dev(o2_last_hour)
if o2_variance < 0.2:
    o2_score = 25
elif o2_variance < 0.5:
    o2_score = 20 - (o2_variance - 0.2) × 16.7
else:
    o2_score = 10

# 2. Combustion Completeness Score (0-30 points)
if CO_avg < 50 ppm:
    co_score = 30
elif CO_avg < 200 ppm:
    co_score = 25 - (CO_avg - 50) × 0.033
else:
    co_score = 5

# 3. Efficiency Score (0-25 points)
eff_deviation = abs(efficiency_current - efficiency_baseline)
if eff_deviation < 1.0:
    eff_score = 25
elif eff_deviation < 3.0:
    eff_score = 20 - (eff_deviation - 1.0) × 7.5
else:
    eff_score = 5

# 4. Emissions Score (0-20 points)
nox_margin = (nox_limit - nox_current) / nox_limit
if nox_margin > 0.3:
    nox_score = 20
elif nox_margin > 0.1:
    nox_score = 15
else:
    nox_score = 5

# Total Score
combustion_quality_score = o2_score + co_score + eff_score + nox_score

# Fouling Detection
if (o2_current - o2_baseline_at_load) > 1.0 AND load_stable:
    fouling_detected = True
```

**Edge Cases:**
- Load changes → Suspend fouling detection during transients
- Fuel quality changes → Recalibrate baseline
- Multiple burners → Track each burner individually if data available

**Non-Functional Requirements:**
- Accuracy: 90% fouling detection rate (validated against maintenance logs)
- Timeliness: Detect fouling within 24 hours of onset
- Integration: CMMS work order creation via REST API

**Estimated Effort:** 4 weeks (1 combustion diagnostics engineer, 1 ML engineer for pattern recognition)

**Business Justification:** Early fouling detection can prevent 2-5% efficiency loss and avoid unplanned outages costing $50K-$500K per day.

---

### Should-Have Features (P1 - High Priority, Post-MVP)

---

#### Feature 6: Fuel Quality Assessment and Deviation Detection

**User Story:**
```
As a fuel manager,
I want the system to infer fuel heating value and moisture content from combustion products,
So that I can verify supplier fuel quality and detect off-spec deliveries.
```

**Acceptance Criteria:**
- [ ] Estimates fuel HHV from flue gas composition and mass balance
- [ ] Calculates fuel quality deviation from contracted specification
- [ ] Detects fuel switching events (change in flue gas signature)
- [ ] Estimates fuel moisture content from stack moisture
- [ ] Tracks fuel quality consistency score over time
- [ ] Alerts if fuel quality deviates >10% from specification
- [ ] Generates fuel quality reports for supplier accountability

**Methodology:**
```
# Simplified fuel HHV estimation from CO2 concentration

# Carbon balance:
C_fuel = (CO2_measured × flue_gas_flow × 12) / (44 × fuel_flow)

# Hydrogen balance (from H2O in flue gas):
H_fuel = (H2O_measured × flue_gas_flow × 2) / (18 × fuel_flow)

# Estimate HHV from C and H content:
HHV_estimated = (C_fuel × 33.8) + (H_fuel × 144) - (O_fuel × 9.0)  # MJ/kg

# Compare to contracted HHV:
deviation_percent = ((HHV_estimated - HHV_contracted) / HHV_contracted) × 100
```

**Estimated Effort:** 3 weeks (1 fuel chemistry engineer, 1 backend engineer)

**Business Justification:** Fuel cost is 60-80% of total combustion system operating cost. Detecting 5% off-spec fuel on a $50M/year fuel spend saves $2.5M/year.

---

#### Feature 7: Economic Analysis and ROI Tracking

**User Story:**
```
As a plant manager,
I want to see the dollar value of fuel savings and emissions reductions achieved by following FLUEFLOW recommendations,
So that I can justify the investment and demonstrate ROI to management.
```

**Acceptance Criteria:**
- [ ] Calculates current fuel cost (USD/hr) based on fuel flow and fuel price
- [ ] Calculates optimized fuel cost if recommendations are followed
- [ ] Quantifies potential fuel savings (USD/hr and USD/year)
- [ ] Calculates emissions penalty/credit cost (if applicable)
- [ ] Estimates carbon tax impact (USD/year) based on CO2 emissions
- [ ] Tracks cumulative savings achieved vs. baseline
- [ ] Calculates ROI and payback period for optimization actions
- [ ] Generates executive summary reports (monthly/quarterly)

**Calculation:**
```
# Current fuel cost
fuel_cost_current = fuel_flow_kg_hr × fuel_price_usd_kg

# Optimized fuel cost (if efficiency improves by X%)
efficiency_improvement = efficiency_optimal - efficiency_current
fuel_cost_optimized = fuel_cost_current × (1 - efficiency_improvement / 100)

# Fuel savings
fuel_savings_usd_hr = fuel_cost_current - fuel_cost_optimized
fuel_savings_usd_year = fuel_savings_usd_hr × operating_hours_per_year

# Carbon tax cost (if applicable)
carbon_tax_usd_year = CO2_tonnes_year × carbon_tax_rate_usd_tonne

# Total savings
total_savings_usd_year = fuel_savings_usd_year + carbon_tax_savings + emissions_penalty_savings

# ROI
roi_months = implementation_cost_usd / (total_savings_usd_year / 12)
```

**Estimated Effort:** 2 weeks (1 financial analyst, 1 backend engineer)

**Business Justification:** Clear ROI demonstration is critical for sales and customer retention.

---

### Could-Have Features (P2 - Nice to Have, Future Roadmap)

#### Feature 8: Automated Combustion Tuning (Autonomous Control)

**User Story:**
```
As a plant operator,
I want the system to automatically adjust O2 setpoints and damper positions to maintain optimal combustion,
So that I can run the plant efficiently without manual intervention.
```

**Scope:** Closed-loop control with DCS integration (requires extensive safety validation)

**Target:** Version 2.0 (Q2 2027)

---

#### Feature 9: Predictive Emissions Modeling

**User Story:**
```
As an environmental manager,
I want to predict future emissions based on planned load changes and fuel switches,
So that I can ensure compliance before making operational changes.
```

**Scope:** ML-based emissions forecasting

**Target:** Version 1.1 (Q3 2026)

---

#### Feature 10: Multi-Fuel Blend Optimization

**User Story:**
```
As a fuel manager,
I want recommendations on optimal fuel blending (e.g., natural gas + hydrogen, coal + biomass),
So that I can minimize cost while meeting emissions and efficiency targets.
```

**Scope:** Optimization for multi-fuel combustion systems

**Target:** Version 1.2 (Q4 2026)

---

### Won't-Have Features (P3 - Out of Scope)

- Carbon capture and storage (CCS) integration → Defer to specialized CCS agents
- Steam turbine optimization → Covered by GL-019 STEAMMASTER
- Boiler water chemistry → Covered by GL-016 WATERGUARD
- Predictive maintenance for rotating equipment → Covered by GL-013 PREDICTMAINT
- Mobile app → Web-only for MVP

---

## 4. Regulatory & Compliance Requirements

### Regulatory Drivers

| Regulation | Effective Date | Scope | Penalty for Non-Compliance |
|------------|---------------|-------|----------------------------|
| EPA 40 CFR Part 60 (NSPS) | Ongoing | New/modified sources in US | $25K-$50K per day |
| EPA 40 CFR Part 63 (NESHAP) | Ongoing | Hazardous air pollutants (US) | $37,500 per day |
| EU IED 2010/75/EU | Ongoing | Industrial emissions (EU) | €1,500 per day + plant shutdown |
| ISO 50001 (Energy Management) | Voluntary | Global | N/A (certification requirement) |
| Clean Air Act Title V | Ongoing | Major sources (US) | Operating permit revocation |

### Compliance Mapping

| Regulatory Requirement | FLUEFLOW Feature | Validation Method |
|------------------------|------------------|-------------------|
| EPA Method 19 efficiency calculation | Combustion Efficiency Analysis | Validate against 100 manual test cases |
| ASME PTC 4.1 heat loss methodology | Heat Loss Breakdown | Third-party engineering review |
| NOx continuous monitoring | Emissions Compliance Monitoring | Compare to CEMS data (±5% tolerance) |
| CO continuous monitoring | Emissions Compliance Monitoring | Compare to CEMS data (±5% tolerance) |
| Quarterly emissions reporting | Performance Report (quarterly) | EPA/EU IED report format validation |
| 7-year data retention | Data Retention Policy | Audit trail verification |
| Calculation auditability | Provenance Tracking (SHA-256) | Regulatory audit simulation |

### Submission Requirements

**Quarterly Emissions Reports (EPA):**
- Format: PDF or XML
- Content: NOx, CO, SOx mass emissions (lb/MMBtu and tonnes)
- Submission: Via EPA's WebFIRE portal

**Annual Energy Efficiency Reports (ISO 50001):**
- Format: Excel or PDF
- Content: Thermal efficiency trends, energy savings achieved
- Submission: Internal certification audit

**Continuous Monitoring Data (CEMS):**
- Format: CSV (1-hour averages)
- Retention: 7 years
- Availability: On-demand for regulatory inspections

---

## 5. User Experience & Workflows

### Primary User Personas

1. **Combustion Engineer (Primary User)**
   - Technical expertise: High
   - Frequency: Daily use (1-2 hours)
   - Key workflows: Analyze efficiency trends, investigate combustion issues, tune burners
   - Top needs: Detailed diagnostics, root cause analysis, tuning recommendations

2. **Plant Operator (Frequent User)**
   - Technical expertise: Medium
   - Frequency: Continuous monitoring (via control room displays)
   - Key workflows: Monitor real-time status, respond to alerts
   - Top needs: Simple dashboards, clear alerts, actionable instructions

3. **Environmental Compliance Manager (Periodic User)**
   - Technical expertise: Medium (environmental regulations)
   - Frequency: Weekly review, quarterly reporting
   - Key workflows: Check emissions compliance, generate reports
   - Top needs: Compliance status summary, automated reports, violation alerts

4. **Plant Manager (Oversight User)**
   - Technical expertise: Low-Medium
   - Frequency: Monthly reviews
   - Key workflows: Review KPIs, track ROI, approve capital projects
   - Top needs: Executive dashboards, ROI metrics, savings tracking

### Key User Flows

#### Flow 1: Daily Efficiency Review (Combustion Engineer)

```
User logs into FLUEFLOW web app
  ↓
Dashboard shows current efficiency: 87.2% (vs. 89.5% baseline)
  ↓
User clicks "Efficiency Trend" to see 7-day chart
  ↓
User notices efficiency drop started 3 days ago
  ↓
User clicks "Diagnostics" → sees "Burner fouling detected (O2 rising)"
  ↓
User reviews recommendation: "Schedule burner cleaning"
  ↓
User creates CMMS work order (1-click integration)
  ↓
User exports efficiency report for management
```

**Time to complete:** <5 minutes

---

#### Flow 2: Emissions Alert Response (Operator)

```
FLUEFLOW sends SMS alert: "NOx exceeds 90% of limit (Unit 1)"
  ↓
Operator opens FLUEFLOW mobile dashboard
  ↓
Dashboard shows NOx = 0.18 lb/MMBtu (limit = 0.20)
  ↓
Operator sees recommendation: "Increase O2 setpoint by 0.5%"
  ↓
Operator adjusts O2 setpoint in DCS
  ↓
FLUEFLOW confirms NOx drops to 0.16 lb/MMBtu after 10 minutes
  ↓
Alert auto-resolves
```

**Time to resolve:** <15 minutes

---

#### Flow 3: Quarterly Emissions Report (Environmental Manager)

```
User navigates to "Reports" section
  ↓
User selects "Quarterly Emissions Report (Q4 2024)"
  ↓
System generates report with:
  - Total NOx, CO, SOx, CO2 emissions (tonnes)
  - Comparison to permit limits
  - Violations summary (0 violations this quarter)
  - Compliance status: COMPLIANT
  ↓
User downloads PDF report
  ↓
User submits to EPA WebFIRE portal (external)
```

**Time to complete:** <10 minutes (vs. 4 hours manual)

---

### Wireframes

*[Wireframes would be created by UX designer - key screens include:]*

1. **Dashboard:** Real-time efficiency, emissions, alerts
2. **Efficiency Trends:** 7-day/30-day/90-day efficiency and heat loss charts
3. **Emissions Compliance:** NOx/CO/SOx status vs. limits, violation log
4. **Burner Diagnostics:** Combustion quality score, fouling indicators, maintenance recommendations
5. **Optimization Recommendations:** Prioritized actions with expected savings
6. **Reports:** Quarterly emissions, annual efficiency, ROI tracking

---

## 6. Success Criteria & KPIs

### Launch Criteria (Go/No-Go Decision)

- [ ] All P0 features implemented and tested
- [ ] 85%+ unit test coverage achieved
- [ ] Integration tested with ≥3 flue gas analyzer brands (ABB, Siemens, Emerson)
- [ ] Integration tested with ≥2 DCS brands (Honeywell, Emerson)
- [ ] Calculation accuracy validated (±0.1 percentage point vs. EPA Method 19 manual calculations)
- [ ] Security audit passed (Grade A score, no critical vulnerabilities)
- [ ] Performance targets met:
  - <10 second end-to-end latency
  - <100 ms calculation time
  - 99.9% uptime (30-day beta test)
- [ ] Regulatory compliance validated:
  - EPA Method 19 calculations peer-reviewed by combustion engineer
  - ASME PTC 4.1 methodology verified
  - Sample emissions reports accepted by EPA portal (test environment)
- [ ] Documentation complete:
  - API documentation (OpenAPI spec)
  - User guide (combustion engineers)
  - Operator quick reference
  - Administrator manual
- [ ] 10 beta customers successfully using the product (≥30 days each)
- [ ] No critical or high-severity bugs in backlog
- [ ] Customer satisfaction (beta) NPS >40

### Post-Launch Metrics

**30 Days:**
- 50 combustion units live
- 500,000 efficiency calculations performed
- Average efficiency improvement: +2.0 percentage points
- Average NOx reduction: 15%
- <10 support tickets per customer
- 99.9% uptime achieved
- NPS >40

**60 Days:**
- 150 combustion units live
- 2M efficiency calculations performed
- Average efficiency improvement: +2.5 percentage points
- Average NOx reduction: 18%
- $50K average fuel savings per unit (annualized)
- <5 support tickets per customer
- NPS >50

**90 Days:**
- 300 combustion units live
- 5M efficiency calculations performed
- Average efficiency improvement: +3.0 percentage points
- Average NOx reduction: 20%
- $75K average fuel savings per unit (annualized)
- 99.95% uptime
- NPS >60
- $10M ARR achieved

**Year 1 (12 Months):**
- 400 combustion units live
- 20M+ calculations performed
- Average efficiency improvement: +3.0 percentage points sustained
- Average fuel savings: $150K per unit per year
- Total customer savings: $60M/year
- Revenue: $60M ARR ($150K per unit/year)
- Customer retention: >95%
- NPS >70

---

## 7. Roadmap & Milestones

### Phase 1: MVP (Weeks 1-16, Q4 2024 - Q1 2025)

**Week 1-4: Requirements & Architecture**
- Finalize PRD (this document)
- Design system architecture (microservices, event-driven)
- Set up development environment (K8s, CI/CD)
- Establish development standards and coding guidelines

**Week 5-10: Core Development**
- Implement flue gas analyzer integration (Modbus TCP, OPC UA)
- Build combustion efficiency calculation engine (EPA Method 19, ASME PTC 4.1)
- Develop emissions compliance monitoring
- Create air-fuel ratio optimization algorithm
- Build burner performance diagnostics

**Week 11-14: Integration & Testing**
- Integrate with SCADA/DCS systems (OPC UA, Modbus)
- Complete test suite (unit, integration, E2E: 85%+ coverage)
- Conduct load testing (1,000 units simulated)
- Security audit and penetration testing
- Performance optimization

**Week 15-16: Beta & Launch Prep**
- Beta customer onboarding (10 sites)
- User training and documentation
- Beta feedback collection and bug fixes
- Production deployment preparation
- Marketing and sales enablement

**Milestone: MVP Launch (Week 16, Q1 2025)**

---

### Phase 2: Enhancements (Weeks 17-32, Q2-Q3 2025)

**Q2 2025:**
- Fuel quality assessment and deviation detection
- Economic analysis and ROI tracking
- Advanced analytics dashboards
- Multi-fuel support (hydrogen co-firing)
- Predictive emissions modeling (ML-based)

**Q3 2025:**
- Heat recovery opportunity assessment
- Automated combustion tuning (advisory mode)
- Integration with carbon accounting systems
- Mobile app (iOS, Android)
- Multi-site fleet optimization

**Milestone: Phase 2 Launch (Week 32, Q3 2025)**

---

### Phase 3: Scale & Autonomy (Weeks 33-52, Q4 2025 - Q2 2026)

**Q4 2025:**
- Autonomous combustion control (closed-loop)
- Digital twin integration
- Blockchain-based emissions reporting
- Distributed combustion system optimization
- Edge computing deployment option

**Q1-Q2 2026:**
- Advanced CFD-based burner optimization
- Carbon capture readiness
- IoT sensor mesh integration
- Hydrogen-ready combustion systems
- Global regulatory compliance (China, India, Brazil)

**Milestone: Enterprise Ready (Q2 2026)**

---

## 8. Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Regulatory changes (EPA, EU IED)** | Medium | High | - Monitor regulatory announcements quarterly<br>- Modular calculation engine for quick updates<br>- Maintain relationships with regulatory consultants |
| **Low customer adoption** | Medium | High | - Beta program with 10 customers (validate product-market fit)<br>- Gather feedback early and iterate<br>- Clear ROI demonstration ($150K savings/year) |
| **Flue gas analyzer compatibility issues** | High | Medium | - Prioritize top 3 brands (ABB, Siemens, Emerson = 70% market)<br>- Hire industrial integration specialist<br>- Provide generic Modbus/OPC UA fallback |
| **DCS integration complexity** | High | Medium | - Support advisory-only mode (no direct control)<br>- Partner with DCS vendors (Honeywell, Emerson)<br>- Offer professional services for complex integrations |
| **Calculation accuracy disputes** | Low | High | - Achieve ±0.1 percentage point accuracy<br>- Third-party validation by combustion engineering firm<br>- Provide complete provenance tracking (SHA-256 hashes)<br>- Publish white paper on calculation methodology |
| **Performance at scale (1,000+ units)** | Low | High | - Load testing with 1,000 simulated units<br>- Horizontal scaling with Kubernetes<br>- Database partitioning and caching (Redis) |
| **Safety concerns (autonomous control)** | Medium | Critical | - Phase 3 feature only (not MVP)<br>- Extensive safety validation (HAZOP, SIL analysis)<br>- Require customer safety system approval<br>- Advisory-only mode as fallback |
| **Talent shortage (combustion engineers)** | Medium | Medium | - Partner with universities (combustion research groups)<br>- Hire retired combustion engineers as consultants<br>- Knowledge capture and documentation |
| **Customer cybersecurity requirements** | High | Medium | - Achieve IEC 62443 compliance<br>- Air-gapped deployment option<br>- On-premise deployment for sensitive customers |
| **Competitive response (DCS vendors)** | Medium | Medium | - Open integration strategy (not proprietary lock-in)<br>- Partner with DCS vendors (OEM agreements)<br>- Focus on AI differentiation (they lack combustion AI expertise) |

---

## 9. Go-to-Market Strategy

### Pricing Model

**Tiered Subscription (Annual):**

| Tier | Combustion Units | Price per Unit/Year | Total Annual Price | Target Customer |
|------|------------------|---------------------|-------------------|-----------------|
| **Starter** | 1-5 units | $100,000 | $100K-$500K | SME plants, pilot projects |
| **Professional** | 6-20 units | $80,000 | $480K-$1.6M | Mid-size plants |
| **Enterprise** | 21-100 units | $60,000 | $1.26M-$6M | Large plants, refineries |
| **Enterprise Plus** | 100+ units | Custom (volume discount) | $6M+ | Multi-site corporations |

**Pricing Justification:**
- Average fuel savings: $150K per unit/year
- ROI: 1.5:1 to 2.5:1 (depending on tier)
- Payback period: 8-16 months

**Additional Revenue Streams:**
- Professional services (integration, training): $50K-$200K per project
- Premium support (24/7, <1 hr response): +20% of subscription
- Custom development (multi-fuel optimization, etc.): $100K-$500K per feature
- Managed services (GreenLang operates the system): +50% of subscription

### Sales Strategy

**Year 1 Target: $60M ARR (400 units)**

**Phase 1 (Months 1-6): Early Adopters**
- Target: 10 beta customers → 50 paying customers
- Focus: Power plants, refineries with existing relationships
- Sales motion: Direct sales (GreenLang sales team)
- Deal size: $100K-$500K (1-5 units)

**Phase 2 (Months 7-12): Expansion**
- Target: 50 → 400 customers
- Focus: Chemical plants, pulp & paper, food processing
- Sales motion: Direct + channel partners (combustion consultants, DCS vendors)
- Deal size: $500K-$2M (5-20 units)

**Phase 3 (Year 2+): Scale**
- Target: 400 → 2,000 customers
- Focus: Global expansion (EU, Asia-Pacific)
- Sales motion: Channel-led (80% of deals)
- Deal size: $1M-$10M (multi-site enterprises)

### Marketing Channels

1. **Trade Shows & Conferences:**
   - Power-Gen International (Dec 2024)
   - AFRC Combustion Symposium (Sep 2025)
   - ISA Automation Week (Sep 2025)

2. **Content Marketing:**
   - White paper: "EPA Method 19 vs. ASME PTC 4.1: A Practical Comparison"
   - Case studies: 3-5% efficiency improvement at [Customer X]
   - Webinars: "Combustion Optimization 101"

3. **Partnerships:**
   - DCS vendors (Honeywell, Emerson): OEM agreements
   - Combustion consultants: Referral partnerships
   - Flue gas analyzer vendors (ABB, Siemens): Joint marketing

4. **Digital:**
   - SEO: "combustion efficiency software", "NOx compliance monitoring"
   - LinkedIn ads targeting combustion engineers, plant managers
   - YouTube: Product demos, customer testimonials

---

## 10. Technical Architecture (High-Level)

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FLUEFLOW GL-018                          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
           ┌─────▼─────┐                   ┌─────▼─────┐
           │ Data      │                   │ Data      │
           │ Sources   │                   │ Sinks     │
           └───────────┘                   └───────────┘
                 │                               │
    ┌────────────┼────────────┐         ┌────────┼────────┐
    │            │            │         │        │        │
┌───▼───┐  ┌────▼────┐  ┌────▼────┐  ┌─▼──┐  ┌─▼──┐  ┌─▼──┐
│Flue   │  │SCADA/   │  │CEMS     │  │DCS │  │CMMS│  │DB  │
│Gas    │  │DCS      │  │         │  │    │  │    │  │    │
│Analyzer│  │         │  │         │  │    │  │    │  │    │
└───────┘  └─────────┘  └─────────┘  └────┘  └────┘  └────┘
  (Modbus)   (OPC UA)    (OPC UA)    (OPC UA) (REST) (SQL)

                 ┌───────────────────────────┐
                 │   Core Engine (Python)    │
                 ├───────────────────────────┤
                 │ - Efficiency Calculator   │
                 │ - Emissions Analyzer      │
                 │ - Optimizer               │
                 │ - Diagnostics Engine      │
                 │ - Provenance Tracker      │
                 └───────────────────────────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
           ┌─────▼─────┐         ┌─────▼─────┐
           │ FastAPI   │         │ Redis     │
           │ Web Server│         │ Cache     │
           └───────────┘         └───────────┘
                 │
           ┌─────▼─────┐
           │ Grafana   │
           │ Dashboards│
           └───────────┘
```

### Tech Stack

- **Language:** Python 3.11
- **Web Framework:** FastAPI (async, high performance)
- **Calculation Engine:** NumPy, SciPy, Cantera (thermodynamics), CoolProp
- **Industrial Protocols:** asyncua (OPC UA), pymodbus (Modbus TCP)
- **Database:** PostgreSQL (relational), TimescaleDB (time-series)
- **Cache:** Redis (LRU cache, 60-second TTL)
- **Messaging:** RabbitMQ (event-driven)
- **Monitoring:** Prometheus + Grafana
- **Logging:** Structured logging (JSON), centralized (ELK stack)
- **Deployment:** Kubernetes, Docker
- **CI/CD:** GitHub Actions, ArgoCD

---

## 11. Appendices

### Appendix A: Glossary

- **ASME PTC 4.1:** American Society of Mechanical Engineers Performance Test Code for Steam Generating Units
- **CEMS:** Continuous Emissions Monitoring System
- **DCS:** Distributed Control System
- **EPA Method 19:** EPA test method for determining SO2 removal efficiency and emission rates
- **Excess Air:** Air supplied above the stoichiometric requirement for complete combustion
- **HHV:** Higher Heating Value (includes latent heat of water vapor)
- **Lambda:** Air ratio (actual air / stoichiometric air)
- **LHV:** Lower Heating Value (excludes latent heat of water vapor)
- **MCR:** Maximum Continuous Rating (nameplate capacity)
- **NOx:** Nitrogen oxides (NO + NO2)
- **O2 Setpoint:** Target oxygen concentration in flue gas for combustion control
- **OPC UA:** Open Platform Communications Unified Architecture (industrial protocol)
- **ppm @ 3% O2:** Concentration corrected to 3% reference oxygen (dry basis)
- **SCADA:** Supervisory Control and Data Acquisition
- **Stoichiometric Air:** Theoretical minimum air required for complete combustion
- **THC:** Total Hydrocarbons (unburned fuel)

### Appendix B: References

- EPA Method 19: https://www.epa.gov/emc/method-19-so2-and-nox-emissions
- ASME PTC 4.1-2013: https://www.asme.org/codes-standards/find-codes-standards/ptc-4-1-steam-generating-units
- ISO 12039:2019: https://www.iso.org/standard/72555.html
- EU IED 2010/75/EU: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32010L0075
- NFPA 85: https://www.nfpa.org/codes-and-standards/all-codes-and-standards/list-of-codes-and-standards/detail?code=85

### Appendix C: Competitive Product Comparison

| Feature | FLUEFLOW | Honeywell Optimizer | Emerson SmartProcess | Manual Testing |
|---------|----------|-------------------|---------------------|----------------|
| Real-time monitoring | ✓ (5s refresh) | ✓ (10s refresh) | ✓ (15s refresh) | ✗ (quarterly) |
| EPA Method 19 compliant | ✓ | ✓ | ~ (simplified) | ✓ |
| ASME PTC 4.1 compliant | ✓ | ✓ | ✗ | ✓ |
| AI-powered diagnostics | ✓ | ✗ | ✗ | ✗ |
| Multi-brand analyzer support | ✓ (open) | ~ (limited) | ~ (Emerson-only) | N/A |
| Emissions compliance tracking | ✓ | ✓ | ✓ | ✗ |
| Fuel quality assessment | ✓ | ✗ | ✗ | ✗ |
| Economic ROI tracking | ✓ | ~ (basic) | ~ (basic) | ✗ |
| Zero-hallucination guarantee | ✓ | N/A | N/A | N/A |
| Price (per unit/year) | $60K-$100K | $150K-$300K | $100K-$250K | $10K/test |
| Payback period | <18 months | 24-36 months | 24-30 months | N/A |

---

## Approval Signatures

**Product Manager:** ___________________  Date: __________

**Engineering Lead:** ___________________  Date: __________

**Combustion Engineering Lead:** ___________________  Date: __________

**CEO:** ___________________  Date: __________

---

**Document Control:**
- **Version:** 1.0
- **Last Updated:** 2024-12-02
- **Next Review:** 2025-03-02 (quarterly)
- **Owner:** GreenLang Product Team
- **Classification:** Internal - Product Development
