# Product Requirements Document: GL-020 ECONOPULSE

**Agent ID:** GL-020
**Codename:** ECONOPULSE
**Name:** EconomizerPerformanceAgent
**Version:** 1.0
**Date:** 2025-12-03
**Product Manager:** GreenLang Product Management
**Status:** Draft
**Category:** Heat Recovery
**Type:** Monitor
**Complexity:** Low
**Priority:** P2

---

## 1. Executive Summary

### 1.1 Problem Statement

Economizers are critical heat recovery components in power plants, industrial boilers, and process heating systems. They recover waste heat from flue gases to preheat feedwater, improving overall thermal efficiency by 3-5%. However, economizer fouling—caused by soot, ash, and scale deposits—degrades heat transfer performance and costs the industry an estimated **$500M+ annually** in:

- **Excess fuel consumption** due to reduced heat transfer efficiency
- **Unplanned maintenance** from tube failures and corrosion
- **Excessive soot blowing** that wastes steam and erodes tubes
- **Reduced boiler capacity** when fouling restricts gas flow
- **Environmental penalties** from increased emissions per unit output

Current monitoring approaches are reactive and manual:
- Operators rely on periodic inspections during scheduled outages
- Temperature measurements are taken manually and infrequently
- Fouling is detected only after significant performance degradation
- Soot blowing schedules are time-based, not condition-based
- Efficiency losses accumulate silently between inspections

The lack of real-time performance monitoring means fouling-related efficiency losses of 1-3% often go undetected for months, costing a typical 500 MW power plant **$500K-$1.5M annually** in excess fuel costs alone.

### 1.2 Solution Overview

**GL-020 ECONOPULSE** is a real-time economizer performance monitoring agent that continuously tracks heat transfer efficiency, detects fouling trends, and generates actionable cleaning alerts. The agent transforms economizer maintenance from reactive to predictive, enabling:

- **Continuous U-value monitoring** to track heat transfer degradation
- **Fouling factor (Rf) trending** to predict cleaning requirements
- **Condition-based soot blowing** to optimize cleaning frequency
- **Efficiency loss quantification** to justify maintenance investments
- **Automated alerts** when fouling exceeds operational thresholds

ECONOPULSE integrates with existing economizer instrumentation (RTDs, thermocouples, flow meters) and soot blower control systems, requiring minimal additional hardware investment.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Soot Blowing Reduction | 10% reduction in cycles | Compare pre/post soot blowing frequency |
| Efficiency Improvement | 0.5% boiler efficiency gain | Heat balance calculation per ASME PTC 4.3 |
| Alert Accuracy | 95% true positive rate | Validated against inspection findings |
| Annual Savings per Unit | $100,000 minimum | Fuel savings + maintenance reduction |
| Time to Deploy | < 1 week per economizer | From sensor integration to live monitoring |
| Customer Satisfaction (NPS) | > 50 | Quarterly customer surveys |

---

## 2. Market Analysis

### 2.1 Market Opportunity

**Total Addressable Market (TAM): $3 Billion**

The economizer performance monitoring market encompasses all facilities with economizers requiring fouling management:

| Segment | Units Worldwide | Avg. Annual Spend | Market Size |
|---------|-----------------|-------------------|-------------|
| Power Plants (Coal/Gas) | 15,000 | $80,000 | $1.2B |
| Industrial Boilers | 50,000 | $20,000 | $1.0B |
| Pulp & Paper Mills | 5,000 | $60,000 | $0.3B |
| Refineries & Petrochemical | 8,000 | $40,000 | $0.32B |
| Waste Heat Recovery | 10,000 | $15,000 | $0.15B |
| Other (Food, Textile, etc.) | 3,000 | $10,000 | $0.03B |
| **Total** | **91,000** | - | **$3.0B** |

**Serviceable Addressable Market (SAM): $800 Million**

Facilities in developed markets with digital infrastructure and regulatory pressure:
- North America: $300M
- Europe: $280M
- Asia-Pacific (developed): $220M

**Serviceable Obtainable Market (SOM): $30 Million (Year 1)**

Initial target of 300 facilities at $100K average contract value.

### 2.2 Market Drivers

1. **Energy Cost Volatility** - Natural gas prices have increased 40% in 3 years, making efficiency gains more valuable
2. **Carbon Pricing** - EU ETS and emerging carbon markets penalize excess fuel consumption
3. **Aging Infrastructure** - 60% of US power plants are 30+ years old with fouling-prone economizers
4. **Decarbonization Mandates** - ISO 50001 and EPA regulations require continuous efficiency improvement
5. **Digital Transformation** - Industrial IoT adoption enables real-time monitoring at lower costs

### 2.3 Competitive Landscape

| Competitor | Approach | Strengths | Weaknesses | ECONOPULSE Advantage |
|------------|----------|-----------|------------|---------------------|
| Manual Inspection | Periodic visual/thermal | Low cost, familiar | Reactive, infrequent | Continuous, predictive |
| DCS Alarms | High/low temperature | Already installed | No fouling calculation | Quantified efficiency loss |
| Portable Analyzers | Periodic measurement | Detailed analysis | Manual, periodic | Automated, continuous |
| OEM Services | Vendor-specific | Deep equipment knowledge | Expensive, lock-in | Vendor-agnostic, affordable |
| Generic APM Platforms | Asset monitoring | Broad coverage | Not economizer-specific | Purpose-built algorithms |

**Differentiation Strategy:**
- Purpose-built for economizer fouling with ASME PTC 4.3 calculations
- Vendor-agnostic—works with any economizer manufacturer
- Low complexity deployment (< 1 week)
- Quantified ROI through efficiency loss monetization

---

## 3. Target Customers

### 3.1 Primary Customers

**Power Generation (Coal and Natural Gas)**

- **Profile:** 200-1000 MW generating units with tubular economizers
- **Pain Points:** Fuel costs are 60-70% of operating expense; 0.5% efficiency loss = $500K/year
- **Decision Makers:** Plant Manager, Performance Engineer, O&M Director
- **Budget:** $50K-$200K for monitoring solutions
- **Sales Cycle:** 6-12 months

**Pulp and Paper Mills**

- **Profile:** Recovery boilers and power boilers with high fouling rates
- **Pain Points:** Soot from black liquor combustion causes rapid fouling; frequent soot blowing
- **Decision Makers:** Mill Manager, Reliability Engineer, Energy Manager
- **Budget:** $30K-$100K
- **Sales Cycle:** 3-6 months

### 3.2 Secondary Customers

**Refineries and Petrochemical Plants**

- **Profile:** Process heaters and waste heat boilers with economizers
- **Pain Points:** Complex fouling from process contaminants; tube failures
- **Decision Makers:** Reliability Manager, Process Engineer
- **Budget:** $40K-$150K
- **Sales Cycle:** 6-9 months

**Food and Beverage Processing**

- **Profile:** Package boilers with economizers for steam generation
- **Pain Points:** Limited engineering staff; need simple solutions
- **Decision Makers:** Plant Engineer, Facilities Manager
- **Budget:** $15K-$50K
- **Sales Cycle:** 2-4 months

### 3.3 User Personas

#### Persona 1: Boiler Operator (Primary User)

**Name:** Mike Thompson
**Role:** Control Room Operator
**Experience:** 15 years in power plant operations
**Goals:**
- Maintain stable boiler operation within parameters
- Respond to alarms quickly and effectively
- Avoid unplanned shutdowns on his shift

**Pain Points:**
- Too many nuisance alarms from DCS
- No clear guidance on when to initiate soot blowing
- Blamed when efficiency drops but lacks visibility into causes

**Needs from ECONOPULSE:**
- Clear, actionable alerts (not just "temperature high")
- Simple dashboard showing economizer health status
- Recommended actions (e.g., "Initiate soot blowing sequence A")

#### Persona 2: Maintenance Engineer (Key Influencer)

**Name:** Sarah Chen
**Role:** Mechanical Maintenance Engineer
**Experience:** 8 years in boiler maintenance
**Goals:**
- Reduce unplanned maintenance events
- Optimize planned outage work scopes
- Extend equipment life and reduce repair costs

**Pain Points:**
- Economizer inspections require expensive scaffolding
- Tube failures cause forced outages costing $500K+
- Difficult to justify maintenance budgets without data

**Needs from ECONOPULSE:**
- Trending data to predict maintenance requirements
- Evidence to justify cleaning or tube replacement
- Integration with CMMS for work order generation

#### Persona 3: Performance Engineer (Technical Champion)

**Name:** David Park
**Role:** Plant Performance Engineer
**Experience:** 5 years in thermal performance analysis
**Goals:**
- Maximize heat rate and efficiency
- Identify and quantify losses
- Report performance to management

**Pain Points:**
- Manual heat balance calculations take hours
- Difficult to isolate economizer losses from other factors
- Limited real-time visibility into performance

**Needs from ECONOPULSE:**
- Automated ASME PTC 4.3 calculations
- Quantified efficiency loss in $/hour
- Historical trending and benchmarking

#### Persona 4: Plant Manager (Economic Buyer)

**Name:** Jennifer Walsh
**Role:** Plant Manager
**Experience:** 20 years in power generation
**Goals:**
- Meet capacity and availability targets
- Control operating costs
- Comply with environmental regulations

**Pain Points:**
- Efficiency losses erode profit margins
- Maintenance costs are unpredictable
- Board demands ROI justification for investments

**Needs from ECONOPULSE:**
- Executive dashboard with KPIs
- Clear ROI metrics ($/year savings)
- Regulatory compliance documentation

---

## 4. P0 Features (Must-Have for MVP)

### 4.1 Feature: Feedwater and Flue Gas Temperature Monitoring

**User Story:**
```
As a boiler operator,
I want to view real-time feedwater and flue gas temperatures at the economizer,
So that I can verify the economizer is operating within design parameters.
```

**Acceptance Criteria:**
- [ ] Displays feedwater inlet and outlet temperatures (accuracy +/- 0.5 deg C)
- [ ] Displays flue gas inlet and outlet temperatures (accuracy +/- 1.0 deg C)
- [ ] Updates readings every 5 seconds
- [ ] Configurable high/low alarm setpoints per measurement point
- [ ] Historical data retention for minimum 1 year
- [ ] Supports RTD, thermocouple, and 4-20mA input types

**Technical Requirements:**
- Input Range: -50 to 600 deg C
- Resolution: 0.1 deg C
- Data protocols: Modbus TCP, OPC-UA, MQTT
- Sampling rate: 1 Hz minimum

**Estimated Effort:** 1 week (1 backend engineer)

---

### 4.2 Feature: Heat Transfer Coefficient (U-value) Calculation

**User Story:**
```
As a performance engineer,
I want the system to calculate the overall heat transfer coefficient continuously,
So that I can track economizer thermal performance degradation over time.
```

**Acceptance Criteria:**
- [ ] Calculates U-value using Log Mean Temperature Difference (LMTD) method
- [ ] Updates calculation every 1 minute during stable operation
- [ ] Filters calculations during transient conditions (load changes, startup)
- [ ] Compares current U-value to design (clean) baseline
- [ ] Displays U-value degradation as percentage
- [ ] Calculation accuracy within 2% of manual ASME PTC 4.3 calculation

**Technical Calculation (ASME PTC 4.3):**

```
Heat Transfer Rate:
Q = m_water * Cp * (T_water_out - T_water_in)

Where:
- Q = Heat transfer rate (kW)
- m_water = Feedwater mass flow rate (kg/s)
- Cp = Specific heat of water (kJ/kg-K)
- T_water_out = Feedwater outlet temperature (deg C)
- T_water_in = Feedwater inlet temperature (deg C)

Log Mean Temperature Difference:
LMTD = (dT1 - dT2) / ln(dT1 / dT2)

Where:
- dT1 = T_gas_in - T_water_out (hot end temperature difference)
- dT2 = T_gas_out - T_water_in (cold end temperature difference)

Overall Heat Transfer Coefficient:
U = Q / (A * LMTD)

Where:
- U = Overall heat transfer coefficient (W/m2-K)
- A = Economizer heat transfer surface area (m2)
- LMTD = Log mean temperature difference (K)
```

**Edge Cases:**
- If dT1 = dT2, use arithmetic mean instead of LMTD
- If flow rate < 10% of design, skip calculation (insufficient data)
- If temperatures indicate reversed flow, generate alarm

**Estimated Effort:** 2 weeks (1 calculator engineer)

---

### 4.3 Feature: Fouling Factor (Rf) Trending

**User Story:**
```
As a maintenance engineer,
I want to track the fouling factor trend over time,
So that I can predict when cleaning will be required and plan maintenance accordingly.
```

**Acceptance Criteria:**
- [ ] Calculates fouling factor (Rf) from U-value comparison
- [ ] Displays Rf trend over configurable time periods (day, week, month)
- [ ] Predicts days until Rf reaches cleaning threshold (linear extrapolation)
- [ ] Supports separate tracking for gas-side and water-side fouling
- [ ] Exports trend data to CSV for external analysis
- [ ] Displays fouling rate (delta Rf per day) for trending

**Technical Calculation:**

```
Fouling Factor:
Rf = (1 / U_fouled) - (1 / U_clean)

Where:
- Rf = Total fouling resistance (m2-K/W)
- U_fouled = Current overall heat transfer coefficient (W/m2-K)
- U_clean = Clean (design) overall heat transfer coefficient (W/m2-K)

Typical Fouling Factor Ranges:
- Clean economizer: Rf < 0.0001 m2-K/W
- Light fouling: 0.0001 < Rf < 0.0003 m2-K/W
- Moderate fouling: 0.0003 < Rf < 0.0005 m2-K/W
- Heavy fouling: Rf > 0.0005 m2-K/W (cleaning recommended)
```

**Estimated Effort:** 1 week (1 backend engineer)

---

### 4.4 Feature: Economizer Effectiveness Calculation

**User Story:**
```
As a performance engineer,
I want to monitor economizer thermal effectiveness continuously,
So that I can quantify how well the economizer is recovering waste heat.
```

**Acceptance Criteria:**
- [ ] Calculates effectiveness using NTU-effectiveness method
- [ ] Displays current effectiveness vs. design effectiveness
- [ ] Updates every 1 minute during stable operation
- [ ] Accounts for partial load operation (normalized effectiveness)
- [ ] Displays effectiveness loss in percentage points

**Technical Calculation:**

```
Economizer Effectiveness:
epsilon = (T_water_out - T_water_in) / (T_gas_in - T_water_in)

Where:
- epsilon = Economizer effectiveness (dimensionless, 0 to 1)
- T_water_out = Feedwater outlet temperature (deg C)
- T_water_in = Feedwater inlet temperature (deg C)
- T_gas_in = Flue gas inlet temperature (deg C)

Typical Effectiveness Ranges:
- Design (clean): 0.70 - 0.85
- Acceptable: 0.60 - 0.70
- Degraded: 0.50 - 0.60
- Severe fouling: < 0.50
```

**Estimated Effort:** 1 week (1 backend engineer)

---

### 4.5 Feature: Cleaning Alerts (Threshold-Based)

**User Story:**
```
As a boiler operator,
I want to receive clear alerts when the economizer needs cleaning,
So that I can initiate soot blowing or schedule maintenance at the right time.
```

**Acceptance Criteria:**
- [ ] Configurable alert thresholds for Rf, effectiveness, and U-value
- [ ] Three alert levels: Advisory, Warning, Critical
- [ ] Alerts include recommended action (soot blow, inspect, clean)
- [ ] Suppresses nuisance alarms during startup/shutdown
- [ ] Delivers alerts via dashboard, email, and SMS
- [ ] Logs all alerts with timestamps for audit trail

**Default Alert Thresholds:**

| Parameter | Advisory | Warning | Critical |
|-----------|----------|---------|----------|
| Fouling Factor (Rf) | > 0.0003 | > 0.0004 | > 0.0005 |
| Effectiveness Loss | > 5% | > 10% | > 15% |
| U-value Degradation | > 10% | > 15% | > 20% |
| Flue Gas Exit Temp Rise | > 10 deg C | > 20 deg C | > 30 deg C |

**Estimated Effort:** 1 week (1 backend engineer)

---

### 4.6 Feature: Efficiency Loss Quantification

**User Story:**
```
As a plant manager,
I want to see the cost of economizer fouling in dollars per hour,
So that I can make informed decisions about cleaning investments.
```

**Acceptance Criteria:**
- [ ] Calculates fuel penalty from economizer efficiency loss
- [ ] Displays efficiency loss in $/hour, $/day, $/month
- [ ] Configurable fuel cost input ($/MMBtu or $/MWh)
- [ ] Shows cumulative losses since last cleaning
- [ ] Compares losses to estimated cleaning cost for ROI analysis
- [ ] Generates monthly efficiency loss report

**Technical Calculation:**

```
Efficiency Loss from Fouling:
delta_eta = (epsilon_clean - epsilon_current) * (T_gas_in - T_water_in) / T_gas_in

Fuel Penalty:
Fuel_penalty ($/hr) = (delta_eta / eta_boiler) * Heat_input * Fuel_cost

Where:
- delta_eta = Efficiency loss due to fouling (%)
- eta_boiler = Overall boiler efficiency (%)
- Heat_input = Boiler heat input (MMBtu/hr)
- Fuel_cost = Fuel cost ($/MMBtu)

Example:
- 0.5% efficiency loss on 500 MW boiler
- Heat input: 5,000 MMBtu/hr
- Fuel cost: $3/MMBtu
- Fuel penalty: (0.005 / 0.88) * 5000 * 3 = $85/hr = $2,040/day
```

**Estimated Effort:** 1 week (1 backend engineer)

---

## 5. P1 Features (Should-Have)

### 5.1 Feature: Soot Blower Optimization

**User Story:**
```
As an operations supervisor,
I want the system to recommend optimal soot blowing sequences,
So that I can minimize steam consumption while maintaining economizer cleanliness.
```

**Acceptance Criteria:**
- [ ] Monitors soot blower effectiveness (Rf change per cycle)
- [ ] Recommends soot blowing frequency based on fouling rate
- [ ] Identifies ineffective soot blowers for maintenance
- [ ] Tracks steam consumption per soot blowing cycle
- [ ] Calculates optimal blowing interval to minimize total cost

**Estimated Effort:** 2 weeks (1 backend engineer, 1 integration engineer)

---

### 5.2 Feature: Predictive Fouling Alerts

**User Story:**
```
As a maintenance planner,
I want the system to predict when fouling will reach critical levels,
So that I can schedule cleaning during planned outages.
```

**Acceptance Criteria:**
- [ ] Predicts time to threshold using fouling rate trend
- [ ] Adjusts prediction based on operating conditions (load, fuel quality)
- [ ] Integrates with outage planning calendar
- [ ] Provides confidence interval on predictions
- [ ] Alerts 2 weeks before predicted threshold breach

**Estimated Effort:** 3 weeks (1 data scientist, 1 backend engineer)

---

### 5.3 Feature: Tube Leak Detection

**User Story:**
```
As a maintenance engineer,
I want early warning of economizer tube leaks,
So that I can schedule repairs before a forced outage occurs.
```

**Acceptance Criteria:**
- [ ] Monitors feedwater flow vs. steam production for mass balance
- [ ] Detects abnormal temperature patterns indicating leaks
- [ ] Alerts on sudden effectiveness changes (leak signature)
- [ ] Estimates leak rate based on mass balance deviation
- [ ] Distinguishes tube leaks from fouling-related changes

**Estimated Effort:** 2 weeks (1 backend engineer)

---

### 5.4 Feature: Water-Side Scaling Detection

**User Story:**
```
As a water treatment specialist,
I want to detect water-side scaling in the economizer,
So that I can adjust water chemistry before tubes are damaged.
```

**Acceptance Criteria:**
- [ ] Separates water-side fouling from gas-side fouling
- [ ] Correlates with feedwater quality data (hardness, pH, oxygen)
- [ ] Alerts on scaling indicators (temperature profile changes)
- [ ] Recommends water treatment adjustments
- [ ] Tracks scaling trend separately from soot fouling

**Estimated Effort:** 2 weeks (1 backend engineer, 1 domain expert)

---

## 6. Technical Specifications

### 6.1 Calculation Standards

All calculations conform to **ASME PTC 4.3 - Performance Test Code for Air Heaters** (applicable to economizers) and industry-standard heat exchanger analysis methods.

### 6.2 Core Calculations Reference

```
======================================================================
ECONOPULSE CALCULATION ENGINE - TECHNICAL REFERENCE
======================================================================

1. LOG MEAN TEMPERATURE DIFFERENCE (LMTD)
----------------------------------------------------------------------
For counterflow economizer arrangement:

LMTD = (dT1 - dT2) / ln(dT1 / dT2)

Where:
  dT1 = T_gas_in - T_water_out    [Hot end]
  dT2 = T_gas_out - T_water_in    [Cold end]

Special case when dT1 = dT2:
  LMTD = dT1 = dT2 (use arithmetic mean)

Units: deg C or K (consistent throughout)


2. OVERALL HEAT TRANSFER COEFFICIENT (U-VALUE)
----------------------------------------------------------------------
U = Q / (A * LMTD)

Where:
  Q = Heat duty (kW)
    = m_water * Cp * (T_water_out - T_water_in)
  A = Heat transfer surface area (m2)
  LMTD = Log mean temperature difference (K)

Result: U in W/(m2-K) or kW/(m2-K)


3. FOULING FACTOR (Rf)
----------------------------------------------------------------------
Rf = (1 / U_fouled) - (1 / U_clean)

Where:
  U_fouled = Current measured U-value
  U_clean = Design (clean) U-value from equipment datasheet

Result: Rf in m2-K/W

Interpretation:
  Rf < 0.0001     Clean
  0.0001-0.0003   Light fouling
  0.0003-0.0005   Moderate fouling
  > 0.0005        Heavy fouling - cleaning required


4. ECONOMIZER EFFECTIVENESS (epsilon)
----------------------------------------------------------------------
epsilon = (T_water_out - T_water_in) / (T_gas_in - T_water_in)

This represents actual heat transfer divided by maximum possible
heat transfer (if water could be heated to gas inlet temperature).

Result: epsilon dimensionless (typically 0.60 - 0.85)


5. EFFICIENCY LOSS CALCULATION
----------------------------------------------------------------------
Flue gas exit temperature rise due to fouling:

dT_exit = T_gas_out_fouled - T_gas_out_clean

Efficiency loss (Siegert formula approximation):
delta_eta = dT_exit * k

Where k depends on fuel type:
  Natural gas: k = 0.045 %/deg C
  Coal: k = 0.038 %/deg C
  Oil: k = 0.042 %/deg C

Fuel penalty:
Cost ($/hr) = (delta_eta / 100) * Heat_input (MMBtu/hr) * Fuel_price ($/MMBtu)

======================================================================
```

### 6.3 Data Input Requirements

| Parameter | Source | Range | Resolution | Update Rate |
|-----------|--------|-------|------------|-------------|
| Feedwater inlet temp | RTD/Thermocouple | 80-200 deg C | 0.1 deg C | 1 Hz |
| Feedwater outlet temp | RTD/Thermocouple | 150-300 deg C | 0.1 deg C | 1 Hz |
| Flue gas inlet temp | Thermocouple | 250-500 deg C | 1 deg C | 1 Hz |
| Flue gas outlet temp | Thermocouple | 120-250 deg C | 1 deg C | 1 Hz |
| Feedwater flow rate | Flow meter | 0-100% | 0.1% | 1 Hz |
| Boiler load | DCS | 0-100% | 0.1% | 1 Hz |
| Soot blower status | DCS | On/Off | Boolean | On change |

### 6.4 System Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Plant DCS/PLC   |---->|  ECONOPULSE Agent |---->|    Dashboard      |
| (Temperature,     |     | (Calculations,    |     | (Visualization,   |
|  Flow, Status)    |     |  Alerts, Trends)  |     |  Reports, Alerts) |
+-------------------+     +-------------------+     +-------------------+
        |                         |                         |
        v                         v                         v
+-------------------+     +-------------------+     +-------------------+
| Modbus TCP/OPC-UA |     | PostgreSQL DB     |     | REST API / MQTT   |
+-------------------+     +-------------------+     +-------------------+
```

### 6.5 Integration Requirements

**Data Protocols:**
- Modbus TCP (primary)
- OPC-UA (preferred for modern DCS)
- MQTT (for edge deployment)
- REST API (for cloud integration)

**Soot Blower Integration:**
- Read soot blower status (operating/idle)
- Optional: Write commands to initiate soot blowing
- Track steam consumption per cycle

**CMMS Integration:**
- Generate work orders on critical alerts
- Export maintenance history for analysis
- Sync equipment hierarchy

---

## 7. Regulatory and Standards Compliance

### 7.1 Applicable Standards

| Standard | Description | Relevance |
|----------|-------------|-----------|
| ASME PTC 4.3 | Air Heater Performance Test Code | Calculation methodology |
| ASME PTC 4 | Steam Generator Performance | Overall boiler efficiency |
| ISO 50001 | Energy Management Systems | Continuous improvement |
| EPA 40 CFR 60 | New Source Performance Standards | Emissions efficiency |
| IEC 62443 | Industrial Cybersecurity | Data security requirements |

### 7.2 ASME PTC 4.3 Compliance

ECONOPULSE calculations align with ASME PTC 4.3 methodology:

- **Heat transfer calculation:** Uses LMTD method per Section 5
- **Uncertainty analysis:** Propagates instrument uncertainties per Section 7
- **Test conditions:** Filters data during transients per Section 4
- **Reporting:** Generates PTC-compliant performance reports

### 7.3 ISO 50001 Alignment

ECONOPULSE supports ISO 50001 energy management requirements:

- **Energy baseline:** Establishes clean economizer performance baseline
- **Energy performance indicators:** Tracks U-value, effectiveness, Rf as EnPIs
- **Monitoring and measurement:** Continuous automated monitoring
- **Corrective actions:** Alerts drive maintenance actions
- **Continual improvement:** Trend analysis shows efficiency gains over time

### 7.4 EPA Energy Efficiency

ECONOPULSE helps facilities demonstrate efficiency improvements for EPA compliance:

- Quantifies efficiency loss from fouling
- Documents cleaning effectiveness
- Generates reports for regulatory submissions
- Supports Best Available Control Technology (BACT) demonstrations

---

## 8. User Experience

### 8.1 Dashboard Overview

The ECONOPULSE dashboard provides role-appropriate views:

**Operator View:**
- Real-time temperature readings (large format)
- Economizer health status (green/yellow/red)
- Active alerts with recommended actions
- Soot blower status and last cleaning time

**Engineer View:**
- U-value and effectiveness trends
- Fouling factor chart with threshold lines
- Performance comparison to baseline
- Calculation details and raw data

**Manager View:**
- Efficiency loss in $/day
- Cumulative losses since last cleaning
- ROI analysis (cleaning cost vs. losses)
- Fleet-wide economizer comparison

### 8.2 Alert Flow

```
Fouling threshold exceeded
         |
         v
+-------------------+
| Alert Generated   |
| - Severity level  |
| - Affected equip  |
| - Recommended act |
+-------------------+
         |
    +----+----+
    |         |
    v         v
Dashboard   Email/SMS
Notification  Alert
    |         |
    v         v
Operator    On-call
Reviews     Engineer
    |       Notified
    v
Action Taken
(Soot blow, etc.)
    |
    v
Alert Cleared
(Automatic on
improvement)
```

### 8.3 Report Types

1. **Shift Report:** Summary of economizer performance for 8-hour shift
2. **Daily Report:** 24-hour performance with efficiency loss totals
3. **Weekly Trend Report:** Fouling trends and prediction update
4. **Monthly Performance Report:** Full analysis with ROI calculations
5. **Cleaning Effectiveness Report:** Before/after comparison post-cleaning

---

## 9. Success Metrics and KPIs

### 9.1 Primary Success Metrics

| Metric | Target | Measurement | Frequency |
|--------|--------|-------------|-----------|
| Soot Blowing Reduction | 10% fewer cycles | Soot blower run count | Monthly |
| Efficiency Gain | 0.5% improvement | Heat balance calculation | Quarterly |
| Alert Accuracy | 95% true positive | Validated vs. inspection | Quarterly |
| Annual Savings | $100K per unit | Fuel + maintenance savings | Annually |
| Deployment Time | < 1 week | Project completion | Per project |

### 9.2 Customer Success Metrics

| Metric | Target | Measurement | Frequency |
|--------|--------|-------------|-----------|
| Customer Satisfaction (NPS) | > 50 | Survey | Quarterly |
| Customer Retention | > 90% | Contract renewals | Annually |
| Support Tickets | < 3/month/customer | Ticket count | Monthly |
| Feature Adoption | > 80% using alerts | Usage analytics | Monthly |

### 9.3 Business Metrics

| Metric | Year 1 Target | Year 2 Target | Year 3 Target |
|--------|---------------|---------------|---------------|
| Customers | 50 | 150 | 400 |
| ARR | $5M | $15M | $40M |
| Gross Margin | 70% | 75% | 80% |
| Customer Acquisition Cost | $15K | $12K | $10K |
| Lifetime Value | $300K | $350K | $400K |

### 9.4 ROI Model

**Typical Customer ROI (500 MW Power Plant):**

| Benefit Category | Annual Value |
|------------------|-------------|
| Fuel savings (0.5% efficiency) | $75,000 |
| Soot blowing reduction (10%) | $15,000 |
| Avoided tube failures | $30,000 |
| Maintenance optimization | $10,000 |
| **Total Annual Benefit** | **$130,000** |
| ECONOPULSE Annual Cost | $30,000 |
| **Net Annual Savings** | **$100,000** |
| **ROI** | **333%** |
| **Payback Period** | **4 months** |

---

## 10. Product Roadmap

### 10.1 Phase 1: MVP (Q3 2026)

**Timeline:** Weeks 1-12

**Week 1-3: Requirements and Design**
- Finalize PRD and technical specifications
- Design system architecture
- Define data model and API contracts
- Create UI/UX wireframes

**Week 4-7: Core Development**
- Implement temperature monitoring agent
- Build U-value calculation engine
- Develop fouling factor trending
- Create effectiveness calculator
- Build alert engine

**Week 8-10: Integration and Testing**
- DCS/PLC integration (Modbus, OPC-UA)
- Dashboard development
- Unit and integration testing
- Performance testing with simulated data

**Week 11-12: Beta and Launch**
- Deploy to 3 beta customers
- Gather feedback and iterate
- Documentation and training materials
- MVP launch

**MVP Deliverables:**
- All P0 features operational
- Dashboard for operators and engineers
- Basic alerting (threshold-based)
- Integration with 2+ DCS platforms
- User documentation

### 10.2 Phase 2: Enhancements (Q4 2026)

**Timeline:** Weeks 13-24

**Features:**
- Soot blower optimization (P1)
- Predictive fouling alerts (P1)
- Tube leak detection (P1)
- Water-side scaling detection (P1)
- Mobile app for alerts
- CMMS integration (SAP PM, Maximo)

**Improvements:**
- Machine learning for fouling prediction
- Fleet-wide benchmarking
- Advanced reporting and analytics
- Multi-language support (German, Spanish, Chinese)

### 10.3 Phase 3: Scale (2027)

**Q1 2027:**
- Air heater monitoring (extension of economizer logic)
- Superheater/reheater fouling detection
- Full boiler performance suite

**Q2 2027:**
- Edge deployment option (on-premise)
- Advanced cybersecurity features (IEC 62443)
- Third-party integrations (OSIsoft PI, Aveva)

**Q3-Q4 2027:**
- Autonomous cleaning optimization (closed-loop control)
- Digital twin integration
- Carbon accounting integration

---

## 11. Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Sensor data quality issues | High | Medium | Data validation, outlier detection, quality scores |
| DCS integration complexity | Medium | High | Pre-built connectors, integration specialists |
| Customer adoption resistance | Medium | Medium | ROI calculator, pilot programs, case studies |
| Calculation accuracy questions | Low | High | ASME PTC 4.3 compliance, third-party validation |
| Competition from DCS vendors | Medium | Medium | Vendor-agnostic positioning, rapid innovation |
| Cybersecurity concerns | Medium | High | IEC 62443 compliance, security audits |

---

## 12. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| ASME PTC | American Society of Mechanical Engineers Performance Test Code |
| DCS | Distributed Control System |
| Economizer | Heat exchanger that preheats feedwater using flue gas waste heat |
| Fouling | Accumulation of deposits (soot, scale) on heat transfer surfaces |
| LMTD | Log Mean Temperature Difference |
| NTU | Number of Transfer Units (heat exchanger sizing method) |
| Rf | Fouling factor/fouling resistance |
| RTD | Resistance Temperature Detector |
| Soot Blower | Device that uses steam jets to clean deposits from tubes |
| U-value | Overall heat transfer coefficient |

### Appendix B: References

1. ASME PTC 4.3-2017: Air Heaters
2. ASME PTC 4-2013: Fired Steam Generators
3. ISO 50001:2018: Energy Management Systems
4. TEMA Standards (Tubular Exchanger Manufacturers Association)
5. EPA 40 CFR Part 60: Standards of Performance for New Stationary Sources

### Appendix C: Emission Factor Sources

| Source | Coverage | Update Frequency |
|--------|----------|------------------|
| EPA AP-42 | US emission factors | Annual |
| IEA | International energy data | Annual |
| EIA | US energy statistics | Monthly |

---

## Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Manager | | | |
| Engineering Lead | | | |
| Domain Expert | | | |
| CEO | | | |

---

**Document Control:**
- Created: 2025-12-03
- Last Modified: 2025-12-03
- Next Review: 2026-01-03
